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
            # Required by SkyPilot's Kubernetes/SSH-node-pool bootstrap:
            #   - socat + nc: portforward networking mode needs both.
            #     Use netcat-gnu (not plain `netcat`/libressl) — SkyPilot runs
            #     `nc -h` and treats non-zero exit as "not installed";
            #     libressl's nc exits 1 on -h, GNU netcat exits 0.
            #   - kubectl: SkyPilot installs k3s on SSH nodes and talks to it
            #     via kubectl.
            socat
            netcat-gnu
            kubectl
            # Playwright browser automation — python package + NixOS-provided browsers.
            # The nixpkgs python package is patched to use store paths for the node
            # driver, so it works on NixOS without nix-ld.  playwright-driver.browsers
            # includes chromium + headless_shell (required by default launch()).
            python312Packages.playwright
            playwright-driver.browsers
            # LaTeX toolchain for building the papers/ directory.
            # scheme-medium does NOT include IEEEtran; use a custom combination that
            # extends scheme-medium with IEEEtran and a few useful extras.
            (texlive.combine {
              inherit (texlive)
                scheme-full
                ieeetran
                biblatex
                biber
                cm-super
                cmap
                latexmk
                # QD_2026.cls requirements not in scheme-medium
                footmisc
                moresize
                diagbox
                eepic
                apacite
                nimbus15
                anyfontsize
                appendix;
            })
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
