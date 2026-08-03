project(
  'maffe',
  'cpp',
  meson_version: '>= 1.3.0',
  version: '0.1',
  license: 'MIT',
  default_options: ['warning_level=3', 'cpp_std=c++23', 'b_lto=true'],
)

cmake = import('cmake')
cpp = meson.get_compiler('cpp')
competition = get_option('competition')
python_enabled = get_option('python')
gurobi = get_option('gurobi')
subproject_lto = competition and get_option('b_lto')
py = disabler()
py_dep = dependency('', required: false)
pybind11_dep = dependency('', required: false)
if python_enabled
  if competition
    error('python is unavailable with competition=true')
  endif
  python_mod = import('python')
  py = python_mod.find_installation('python3')
  py_dep = py.dependency()
  pybind11_dep = dependency('pybind11')
endif

highs_options = cmake.subproject_options()
highs_options.set_install(false)
if competition
  highs_options.set_install(true, target: 'highs')
endif
highs_options.add_cmake_defines(
  {
    'BUILD_SHARED_LIBS': 'OFF',
    'BUILD_CXX_EXE': 'OFF',
    'BUILD_EXAMPLES': 'OFF',
    'BUILD_TESTING': 'OFF',
    'ZLIB': 'OFF',
  },
)
if subproject_lto
  highs_options.add_cmake_defines(
    {
      'CMAKE_INTERPROCEDURAL_OPTIMIZATION': 'ON',
    },
  )
endif

highs_project = cmake.subproject('highs', options: highs_options)
highs_dep = highs_project.dependency('highs')
highs_source_dir = meson.project_source_root() / 'subprojects' / 'highs'
highs_build_dir = meson.current_build_dir() / 'subprojects' / 'highs' / '__CMake_build'

gurobi_enabled = false
gurobi_deps = []
if not gurobi.disabled()
  gurobi_header = cpp.has_header('gurobi_c.h', required: gurobi)
  gurobi_c_dep = cpp.find_library('gurobi', required: gurobi)
  if gurobi_header and gurobi_c_dep.found()
    gurobi_enabled = true
    gurobi_deps = [gurobi_c_dep]
  endif
endif

conicbundle_enabled = false
conicbundle_deps = []
if competition
  conicbundle_dep = dependency(
    'conicbundle',
    required: true,
    fallback: ['conicbundle', 'conicbundle_dep'],
  )
  conicbundle_enabled = true
  conicbundle_deps = [conicbundle_dep]
endif

scip_lps = gurobi_enabled ? 'grb' : 'highs'
scip_options = cmake.subproject_options()
scip_options.set_install(false)
if competition
  scip_options.set_install(true, target: 'libscip')
  scip_options.set_install(true, target: 'libsoplex')
endif
scip_cmake_defines = {
  'AUTOBUILD': 'off',
  'AMPL': 'off',
  'BUILD_TESTING': 'off',
  'BOOST': 'off',
  'CMAKE_POSITION_INDEPENDENT_CODE': 'ON',
  'EXACTSOLVE': 'off',
  'GCG': 'off',
  'IPOPT': 'off',
  'LPS': scip_lps,
  'NO_EXTERNAL_CODE': 'on',
  'PAPILO': 'off',
  'READLINE': 'off',
  'SOPLEX': 'on',
  'SHARED': 'off',
  'SYM': 'none',
  'THREADSAFE': 'off',
  'TPI': 'none',
  'UG': 'off',
  'ZLIB': 'off',
  'ZIMPL': 'off',
}
if scip_lps == 'highs'
  scip_options.append_compile_args('cpp', '-I' + highs_source_dir, target: 'libscip')
  scip_options.append_compile_args('cpp', '-I' + (highs_source_dir / 'highs'), target: 'libscip')
  scip_options.append_compile_args('cpp', '-I' + highs_build_dir, target: 'libscip')
  scip_cmake_defines += {
    'HIGHS_DIR': meson.project_source_root() / 'cmake' / 'highs-meson-package',
  }
endif
if subproject_lto
  scip_cmake_defines += {
    'LTO': 'on',
  }
endif
scip_options.add_cmake_defines(scip_cmake_defines)

scipoptsuite_project = cmake.subproject('scipoptsuite', options: scip_options)
scip_dep = scipoptsuite_project.dependency('libscip')
soplex_dep = scipoptsuite_project.dependency('libsoplex')

dependencies = [
  scip_dep,
  soplex_dep,
  highs_dep,
] + gurobi_deps + conicbundle_deps

git_prog = find_program('git', required: false)
git_hash = 'unknown'
git_dirty = 0
if git_prog.found()
  git_hash_cmd = run_command(
    git_prog,
    '-C', meson.project_source_root(),
    'rev-parse',
    'HEAD',
    check: false,
  )
  if git_hash_cmd.returncode() == 0
    git_hash = git_hash_cmd.stdout().strip()
  endif

  git_status_cmd = run_command(
    git_prog,
    '-C', meson.project_source_root(),
    'status',
    '--porcelain',
    check: false,
  )
  if git_status_cmd.returncode() == 0 and git_status_cmd.stdout().strip() != ''
    git_dirty = 1
  endif
endif

config_data = configuration_data()
config_data.set('MAFFE_COMPETITION_BUILD', competition ? 1 : 0)
config_data.set('MAFFE_HAVE_GUROBI', gurobi_enabled ? 1 : 0)
config_data.set('MAFFE_HAVE_CONICBUNDLE', conicbundle_enabled ? 1 : 0)
config_data.set_quoted('MAFFE_PROJECT_VERSION', meson.project_version())
config_data.set_quoted('MAFFE_GIT_HASH', git_hash)
config_data.set('MAFFE_GIT_DIRTY', git_dirty)
config_data.set_quoted('MAFFE_SCIP_VERSION', scip_dep.version())
config_data.set_quoted('MAFFE_SCIP_LPS', scip_lps)
config_data.set_quoted('MAFFE_HIGHS_VERSION', highs_dep.version())

configure_file(
  output: 'config.hpp',
  configuration: config_data,
)

core_sources = [
  'src/branchandprice/highs/root_lp.cpp',
  'src/branchandprice/scip/branch_and_price.cpp',
  'src/branchandprice/pricer/lagrangian.cpp',
  'src/branchandprice/master/set_packing_heuristic.cpp',
  'src/branchandprice/master/set_packing_scip.cpp',
  'src/compact/compact_solver.cpp',
  'src/heuristic/colgen_heuristic.cpp',
  'src/heuristic/node_coverage.cpp',
  'src/maffe/maffe.cpp',
  'src/branchandprice/master/root_master.cpp',
  'src/reductions/chain_rule.cpp',
  'src/reductions/cherry_picking.cpp',
  'src/reductions/cluster_reduction.cpp',
  'src/reductions/cut_aware_cluster.cpp',
  'src/reductions/five_three_reduction.cpp',
  'src/reductions/orchestrator.cpp',
  'src/reductions/three_two_reduction.cpp',
  'src/newick/newick.cpp',
]

include_dirs = include_directories('.', 'include', 'src')

cli_sources = [
  'src/cli/main.cpp',
]

cli_link_args = []
if competition
  cli_link_args += [
    '-static',
  ]
endif

competition_cli_targets = [
  ['maffe-exact', '-1', '0', '0'],
  ['maffe-heuristic', '285', '1', '0'],
  ['maffe-lowerbound', '-1', '0', '1'],
]

core_lib = static_library(
  'maffe-core',
  core_sources,
  include_directories: include_dirs,
  dependencies: dependencies,
  pic: not competition,
  install: false,
  build_by_default: not competition,
)
if competition
  foreach target : competition_cli_targets
    target_cpp_args = [
      '-DMAFFE_COMPETITION_TIMEOUT_SECONDS=' + target[1],
      '-DMAFFE_COMPETITION_HEURISTIC_BUILD=' + target[2],
      '-DMAFFE_COMPETITION_LOWERBOUND_BUILD=' + target[3],
    ]
    target_core_lib = static_library(
      target[0] + '-core',
      core_sources,
      include_directories: include_dirs,
      dependencies: dependencies,
      pic: false,
      cpp_args: target_cpp_args,
      install: false,
    )
    executable(
      target[0],
      cli_sources,
      include_directories: include_dirs,
      dependencies: dependencies,
      link_with: target_core_lib,
      link_args: cli_link_args,
      cpp_args: target_cpp_args,
      install: true,
      install_tag: 'competition',
    )
  endforeach
else
  executable(
    'maffe',
    cli_sources,
    include_directories: include_dirs,
    dependencies: dependencies,
    link_with: core_lib,
    link_args: cli_link_args,
    install: true,
  )
  if python_enabled
    py.extension_module(
      'maffe',
      'bindings/python/module.cpp',
      include_directories: include_dirs,
      dependencies: dependencies + [py_dep, pybind11_dep],
      link_with: core_lib,
      install: true,
    )
  endif
endif
