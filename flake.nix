{
  description = "Python project with uv";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        python = pkgs.python312;
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = with pkgs; [
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
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              uv venv
            fi
            source .venv/bin/activate
            # Set LD_LIBRARY_PATH to find C++ standard library and other native libraries
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:${pkgs.graphviz}/lib:$LD_LIBRARY_PATH"
            export PKG_CONFIG_PATH="${pkgs.graphviz}/lib/pkgconfig:$PKG_CONFIG_PATH"
            echo "Python $(python --version) with uv $(uv --version)"
          '';
        };
      });
}
