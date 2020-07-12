load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

git_repository(
   name = "org_openmined_psi",
   remote = "https://github.com/OpenMined/PSI",
   commit = "2110b0c78aaf535f0f154541a5239ae1d4876023",
   init_submodules = True,
)

load("@org_openmined_psi//private_set_intersection:preload.bzl", "psi_preload")

psi_preload()

load("@org_openmined_psi//private_set_intersection:deps.bzl", "psi_deps")

psi_deps()
