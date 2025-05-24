{
  description = "Development Enviroment";

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    # To update, see https://hydra.nix-community.org/eval/71410#tabs-inputs
    nixpkgs.url = "github:NixOS/nixpkgs/c3ee76c437067f1ae09d6e530df46a3f80977992";
  };

  outputs = { self, flake-utils, nixpkgs }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
            cudaSupport = true;
          };
        };
      in {
        devShells.default = (import ./shell.nix { inherit pkgs; });
      });
}
