{
  description = "3D Object Reconstruction Using Infrared Images (with Occlusion Handling)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      pkgsFor = system: import nixpkgs { inherit system; };
    in
    {
      devShells = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            numpy
            scipy
            matplotlib
            opencv4
            scikit-image
            trimesh
            torch
            pytest
          ]);
        in
        {
          default = pkgs.mkShell {
            packages = [
              pythonEnv
              pkgs.ruff
            ];

            shellHook = ''
              export PYTHONPATH="$PWD:$PYTHONPATH"
              echo "Infrared 3D Reconstruction environment loaded."
              echo "Run 'python3 test_pipeline.py' to execute the benchmark suite."
            '';
          };
        }
      );

      apps = forAllSystems (system:
        let
          pkgs = pkgsFor system;
          pythonEnv = pkgs.python3.withPackages (ps: with ps; [
            numpy
            scipy
          ]);
          runner = pkgs.writeShellScriptBin "run-ir-benchmark" ''
            export PYTHONPATH="${self}:$PYTHONPATH"
            exec ${pythonEnv}/bin/python3 ${self}/test_pipeline.py "$@"
          '';
        in
        {
          default = {
            type = "app";
            program = "${runner}/bin/run-ir-benchmark";
          };
        }
      );
    };
}

