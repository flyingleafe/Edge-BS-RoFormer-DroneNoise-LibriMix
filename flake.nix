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
          ];

          shellHook = ''
            if [ ! -d .venv ]; then
              uv venv
            fi
            source .venv/bin/activate
            # Set LD_LIBRARY_PATH to find C++ standard library and other native libraries
            export LD_LIBRARY_PATH="${pkgs.stdenv.cc.cc.lib}/lib:${pkgs.zlib}/lib:$LD_LIBRARY_PATH"
            echo "Python $(python --version) with uv $(uv --version)"
          '';
        };
      });
}
