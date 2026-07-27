{
  description = "Python project with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { self, nixpkgs, flake-utils, git-hooks }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;

        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./. ;
          hooks = {
            ruff = {
              enable = true;
              # ruff is a placeholder in git-hooks.nix; provide real package
              package = pkgs.ruff;
            };
            ruff-format = {
              enable = true;
              package = pkgs.ruff;
            };
            pyright = {
              enable = true;
            };
            # Syntax-check every YAML file (Hydra configs included). Uses the
            # pre-commit-hooks PyYAML, independent of the project .venv.
            check-yaml = {
              enable = true;
            };
            validate-experiment-docs = {
              enable = true;
              name = "experiment docs";
              entry = "${python}/bin/python scripts/validate_experiment_docs.py";
              # Whole-set contract (not per-file): run once whenever any
              # experiment config or experiment doc changes.
              files = "^(conf/experiment/.*\\.(yaml|md)|docs/experiments/.*\\.md|scripts/validate_experiment_docs\\.py)$";
              pass_filenames = false;
              language = "system";
            };
          };
        };
      in
      {
        # Run hooks with `nix fmt`
        formatter =
          let
            inherit (pre-commit-check.config) package configFile;
            script = ''
              ${pkgs.lib.getExe package} run --all-files --config ${configFile}
            '';
          in
          pkgs.writeShellScriptBin "pre-commit-run" script;

        # Run hooks sandboxed with `nix flake check`
        checks = {
          inherit pre-commit-check;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = pre-commit-check.enabledPackages ++ (with pkgs; [
            python
            uv
            # C++ standard library for NumPy and other native dependencies
            stdenv.cc.cc.lib
            # Additional libraries commonly needed by Python packages
            zlib
            libffi
            # Graphviz for pygraphviz
            graphviz
            pkg-config
            # Playwright browser automation — python package + NixOS-provided browsers.
            # The nixpkgs python package is patched to use store paths for the node
            # driver, so it works on NixOS without nix-ld.  playwright-driver.browsers
            # includes chromium + headless_shell (required by default launch()).
            python312Packages.playwright
            playwright-driver.browsers
            # LaTeX toolchain for writing/papers: Tectonic — a self-contained
            # modern engine (XeTeX core) that fetches packages on demand into
            # its own cache, replacing the multi-GB texlive.combine set.
            # Build: `tectonic main.tex` (or `tectonic -X compile`); biblatex
            # workflows use `tectonic -X build` with a Tectonic.toml, classic
            # bibtex ones just work via the automatic rerun logic.
            tectonic
	    # easier latex for easier docs
	    typst
            # for looking at resulting pdfs
            poppler-utils
	    # unavoidable js
	    nodejs
          ]);

          shellHook = ''
            ${pre-commit-check.shellHook}
            if [ ! -d .venv ]; then
              uv venv
            fi
            source .venv/bin/activate
            # Set LD_LIBRARY_PATH to find C++ standard library and other native libraries
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.graphviz}/lib:$LD_LIBRARY_PATH"
            export PKG_CONFIG_PATH="${pkgs.graphviz}/lib/pkgconfig:$PKG_CONFIG_PATH"
            # Point Playwright at the NixOS-provided browsers (chromium, headless_shell, ffmpeg).
            export PLAYWRIGHT_BROWSERS_PATH="${pkgs.playwright-driver.browsers}"
            # Skip host-requirements validation — NixOS browsers are already patched.
            export PLAYWRIGHT_SKIP_VALIDATE_HOST_REQUIREMENTS=true
            echo "Python $(python --version) with uv $(uv --version)"
          '';
        };
      });
}
