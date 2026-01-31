:- use_module('./scpl_client.pl').
:- use_module(library(reif)).
:- use_module(library(clpz)).

init_env(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.init_env', [Env_py], Res),
    Res = "true".

collect_wood(Env) :- 
    to_py_string([Env], [Env_py]),
    call_py_func('prolog_call_tools.call_wood', [Env_py], Res),
    Res = "true".

run(Env_name) :-
    py_using("prolog_call_tools"),
    init_env(Env_name),
    collect_wood(Env_name).

to_py_string([], []).
to_py_string(Ls_string, Ls_py_string) :-
    Ls_string = [Head_string | Ls_rest_string],
    Ls_py_string = [Head_py_string | Ls_rest_py_string],
    string_to_python_string(Head_string, Head_py_string),
    to_py_string(Ls_rest_string, Ls_rest_py_string).
