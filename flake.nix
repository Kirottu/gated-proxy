{
  description = "Resource gated LLM proxy";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { nixpkgs, self }:
    let
      fsys =
        f:
        nixpkgs.lib.attrsets.genAttrs [
          "x86_64-linux"
          "armv7l-linux"
          "aarch64-linux"
          "x86_64-darwin"
          "aarch64-darwin"
        ] (s: f s);
    in
    {
      packages = fsys (
        a:
        let
          pkgs = nixpkgs.legacyPackages.${a};
          lib = pkgs.lib;

          cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);
          pname = cargoToml.package.name;
          version = cargoToml.package.version;

          pkg = pkgs.rustPlatform.buildRustPackage {
            inherit pname version;

            src = builtins.path {
              path = lib.sources.cleanSource self;
              name = "${pname}-${version}";
            };

            strictDeps = true;

            cargoLock = {
              lockFile = ./Cargo.lock;
            };

            nativeBuildInputs = with pkgs; [
              rustc
              cargo
            ];

            CARGO_BUILD_INCREMENTAL = "false";
            RUST_BACKTRACE = "full";

            meta = {
              description = "A proxy for local LLM use that gates calls based on system conditions";
              mainProgram = "gated-proxy";
              license = [ lib.licenses.mit ];
            };
          };

        in
        {
          gated-proxy = pkg;
          default = pkg;
        }
      );
      devShells = fsys (
        a:
        let
          pkgs = nixpkgs.legacyPackages.${a};
        in
        {
          default = pkgs.mkShell {
            inputsFrom = builtins.attrValues self.packages.${a};
            packages = with pkgs; [
              rustc
              cargo
              clippy
              rustfmt
            ];
          };
        }
      );

    };
}
