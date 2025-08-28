#--------------------------------------------------------------------------------------------------------------------------------
## Requisitos
#--------------------------------------------------------------------------------------------------------------------------------
import Base: +, -
import ITensors: inner, normalize
import TensorMixedStates: Limits, maxlinkdim, State, apply, approx_W, measure, norm, tdvp, truncate
import Roots: find_zero

#--------------------------------------------------------------------------------------------------------------------------------
## Constantes y overhauls básicos
#--------------------------------------------------------------------------------------------------------------------------------

+(v1::State,v2::State;limits::Limits=Limits())=truncate(State(v1,+(v1.state,v2.state;alg="directsum"));limits)
-(v1::State,v2::State;limits::Limits=Limits())=+(v1,-v2;limits=limits)
inner(v1::State,v2::State)=inner(v1.state,v2.state)

# Igual al truncate normal, pero devuelve tambien los errores de cutoff que aparecieron
function truncate_err(state::State; limits::Limits)
    errors = zeros(length(state.system))
    function callback(; link, truncation_error)
        bond_no = last(link)
        errors[bond_no] = truncation_error
        return nothing
    end
    return State(state.type, state.system, truncate(state.state; limits.cutoff, limits.maxdim,callback)), errors
end


function null(type) # función para poner Union{Nothing, type} en los argumentos default
    return Union{Nothing, type}
end

const null_method(dt::AbstractFloat;kwargs...)=((Ψ::State)->(0*Ψ))

#--------------------------------------------------------------------------------------------------------------------------------
## Funciones para la evolución de estados
#--------------------------------------------------------------------------------------------------------------------------------

"""
TODO:   Check implementation of algorithms WI and WII
        Using two different methods for adaptive time step.
        
"""

"""
    step_estimation(steps::Int,dist::Float64,tol::Float64,method_order::Int,cutoff=nothing)::Int

Estimates the number of steps to try for the next time interval of the simulation. If cutoff is given, then it is taken into account to prevent an excessive number of
steps being chosen. If cutoff is not provided, only the trotterization error is taken into account.
Obs: The method targets the minimum number of steps such that err(step)<tol*3/4. 3/4 is chosen to ensure succes on the next try.
The step is not permited to be raised or lowered by more than a factor of 2.
"""
function step_estimation(steps::Int,dist::Float64,tol::Float64,method_order::Int,cutoff=nothing)::Int
    if isnothing(cutoff)
        # nuevo núemro de pasos, estimado para que el error sea justo menor a la tolerancia
        # pero tampoco permite que aumente el número de pasos una barbaridad.
        st = (dist/tol/3*4)^(1/method_order)*steps
        return Int(ceil(min(2*steps, max(st,steps/2))))
    end
    smin = (method_order * steps^method_order * (dist/cutoff - steps))^(1/(method_order+1))
    err = s -> (tol*3/4 - cutoff*s*(1+(smin/s)^(method_order+1) / method_order))
    if err(smin) <= 0
        return Int(ceil(min(2*steps, max(smin, steps/2))))
    end
    k = 1
    while err(smin/2^k) > 0
        k += 1
    end
    st = find_zero(err, (smin/2^k, 2^(k-1)), A42(), xatol=1e-3)
    return Int(ceil(min(2*steps, max(st, steps/2))))
end
    

"""
    default_error(Ψ_1::State,Ψ_2::State,tol::Float64,steps,method_order::Int)

Funcion que compara dos estados obtenidos con distinta precisión, extrayendo una medida de error.
Si esta supera a `tol`, el paso no se aceptará. Si no lo hace, el paso se aceptará.
La función también intenta estimar cual es el número de pasos necesario para que el error sea similar a `tol`
"""
function default_error(Ψ_1::State, Ψ_2::State, tol::Float64, steps::Int, method_order::Int; kwargs...)
    verbose = get(kwargs, :verbose, false)

    dist = (1 - real(inner(Ψ_1.state, Ψ_2.state))) * 2 #(tienen norma uno ergo distancia es 2 - 2*re(inner). Tomo abs para evitar imaginarios)
    accept = dist < tol # Se fija si aceptar el paso

    steps = step_estimation(steps, dist, tol, method_order)
    if verbose
        println("Error de $(round(dist, sigdigits=2)) comparado a la tolerancia de $(round(tol, sigdigits=2)). Se $(accept ? "acepta" : "rechaza") el paso temporal. \n Se estima necesitar $(steps) pasos para controlar el error.")
    end
    return accept, steps
end

"""
    TEBD2_error(Ψ_1::State,Ψ_2::State,tol::Float64,steps::Int,method_order::Int;verbose=false)

Funcion que compara dos estados obtenidos con distinta precisión especializada para el método TEBD2, extrayendo una medida de error.
Si esta supera a `tol`, el paso no se aceptará. Si no lo hace, el paso se aceptará.
La función también intenta estimar cual es el número de pasos necesario para que el error sea similar a `tol`
"""
function TEBD2_error(Ψ_1::State, Ψ_2::State, tol::Float64, steps::Int, method_order::Int; err1=nothing, err2=nothing, kwargs...)
    verbose = get(kwargs, :verbose, false)
    cutoff = get(kwargs, :cutoff, 0.)
    Ufuncs = get(kwargs, :Ufuncs, [])

    l = length(Ufuncs)
    n = length(Ψ_1.system)

    if isnothing(err1)
        cutofferror_1 = 2 * l * n * cutoff * steps # Estima el máximo error producido por cutoffs al evolucionar Ψ_1 (Chequear que esté en lo correcto)
    else
        cutofferror_1 = sum(err1)
    end
    if isnothing(err2)
        cutofferror_2 = 2 * 2 * l * n * cutoff * steps # Estima el máximo error producido por cutoffs al evolucionar Ψ_2 (asumo que tuvo el doble de pasos temporales)
    else
        cutofferror_2 = sum(err2)
    end
    cutoff_err = cutofferror_1 + cutofferror_2

    dist = (1 - real(inner(Ψ_1.state, Ψ_2.state))) * 2 #(tienen norma uno ergo distancia es 2 - 2*re(inner). Tomo abs para evitar imaginarios)
    accept = dist < tol # Se fija si aceptar el paso

    # El error será eps=eps_t+eps_cutoff, en donde eps_cutoff es acotado por cutofferror_1+cutofferror_2
    if tol < cutoff_err
        printstyled("El error deseado es cercano del estimado por el cutoff, habría que bajar el cutoff o aumentar maxdim.\n", color=:red)
        accept = false
    end

    steps = step_estimation(steps, dist, tol, method_order, cutoff_err)

    if verbose
        println("Error de $(round(dist, sigdigits=2)) comparado a la tolerancia de $(round(tol, sigdigits=2)) y cutoff estimado de $(round(cutoff_err, sigdigits=2)). Se $(accept ? "acepta" : "rechaza") el paso temporal. \n Se estima necesitar $(steps) pasos para controlar el error.")
    end
    return accept, steps
end

"""
    TEBD2(Ufuncs::Array{Function}, dt::Float64, Ψ::State, cutoff=0., maxdim=typemax(Int))

Given a time interval dt, and an array of functions which generate the evolution operators corresponding to the internally conmuting parts of the hamiltonian,
returns a function that performs TEBD2 evolution of a given state, and applies a cutoff procedure to the resulting state. 
"""
function TEBD2(Ufuncs::Array{Function}, dt::Float64; cutoff=0., maxdim=typemax(Int))
    gates = [U(dt/2) for U in Ufuncs]
    append!(gates, reverse(gates))
    limits = Limits(cutoff=cutoff, maxdim=maxdim)
    U = prod(gates)
    return (Ψ::State) -> apply(U, Ψ; limits=limits)
end

"""
    TEBD2_adaptive(Ufuncs::Array{Function}, dt::Float64; cutoff=0., maxdim=typemax(Int))

TEBD2, but optimized for use with an adaptive time step.
"""
function TEBD2_adaptive(Ufuncs::Array{Function}, dt::Float64; cutoff=0., maxdim=typemax(Int))
    gates = [U(dt/2) for U in Ufuncs]
    limits = Limits(cutoff=cutoff, maxdim=maxdim)
    f = (Ψ::State) -> begin
        errors = zeros(length(Ψ.system))
        Ψf = copy(Ψ)
        for U in gates
            Ψf, err = truncate_err(apply(U, Ψ), limits)
            errors += err
        end
        return Ψ, errors
    end
    return f
end

"""
    TEBD4(Ufuncs::Array{Function}, dt::Float64, Ψ::State, cutoff=0., maxdim=typemax(Int))

Given a time interval dt, and an array of functions which generate the evolution operators corresponding to the internally conmuting parts of the hamiltonian,
returns a function that performs TEBD4 evolution of a given state, and applies a cutoff procedure to the resulting state. 
"""
function TEBD4(Ufuncs::Array{Function}, dt::Float64; cutoff=0., maxdim=typemax(Int))
    dt1 = dt/(4-4^(1/3))
    dt2 = dt-4*dt1

    gates1 = [U(dt1/2) for U in Ufuncs]
    append!(gates1, reverse(gates1))
    U1 = prod(gates1)

    gates2 = [U(dt2/2) for U in Ufuncs]
    append!(gates2, reverse(gates2))
    U2 = prod(gates2)

    U = U1 * U1 * U2 * U1 * U1
    limits = Limits(cutoff=cutoff, maxdim=maxdim)

    return (Ψ::State) -> apply(U, Ψ; limits=limits)
end

"""
    evolution_methods(alg::String, kwargs...)

Returns a function that, given a time interval, returns another function `step_method` that calculates the evolution of
a given state according to a specified method `alg`. The algorithm's one-step error order `method_order` is also returned.
`kwargs` are used to create the `step_method` function.
"""
function evolution_methods(alg::String; kwargs...)
    if alg == "TEBD2"
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        Ufuncs = get(kwargs, :Ufuncs, Function[])
        verbose = get(kwargs, :verbose, false)
        if Ufuncs == []
            error("TEBD2 requires providing a valid evolution operator array")
        end
        if verbose && cutoff == 0.
            error("Ojo, correr TEBD2 sin un cutoff es bastante mala idea.")
        end

        step_method = (dt::Float64) -> TEBD2(Ufuncs, dt; cutoff=cutoff, maxdim=maxdim)
        method_order = 2
        return step_method, method_order

    elseif alg == "TEBD2_adaptive"
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        Ufuncs = get(kwargs, :Ufuncs, Function[])
        verbose = get(kwargs, :verbose, false)
        if Ufuncs == []
            error("TEBD2 requires providing a valid evolution operator array")
        end
        if verbose && cutoff == 0.
            error("Ojo, correr TEBD2 sin un cutoff es bastante mala idea.")
        end

        step_method = (dt::Float64) -> TEBD2_adaptive(Ufuncs, dt; cutoff=cutoff, maxdim=maxdim)
        method_order = 2
        return step_method, method_order

    elseif alg == "TEBD4"
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        Ufuncs = get(kwargs, :Ufuncs, Function[])
        verbose = get(kwargs, :verbose, false)
        if Ufuncs == []
            error("TEBD4 requires providing a valid evolution operator array")
        end
        if length(Ufuncs) > 2
            error("TEBD4 not yet implemented for more than 2 internally conmuting parts")
        end
        if verbose && cutoff == 0.
            error("Ojo, correr TEBD4 sin un cutoff es bastante mala idea.")
        end

        step_method = (dt::Float64) -> TEBD2(Ufuncs, dt; cutoff=cutoff, maxdim=maxdim)
        method_order = 4

    elseif alg == "TDVP"
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        evolver = get(kwargs, :evolver, nothing)
        if isnothing(evolver)
            error("TDVP requires providing a valid Hamiltonian!")
        end

        step_method = (dt::Float64) -> ((Ψ::State) -> tdvp(evolver, dt, Ψ; cutoff=cutoff, maxdim=maxdim))
        method_order = 1
        return step_method, method_order

    elseif alg == "WI"
        println("No estoy seguro de que WI esté implementeado correctamente, hay que revisar")
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        order = get(kwargs, :order, nothing)
        evolver = get(kwargs, :evolver, nothing)
        if isnothing(evolver) | isnothing(order) | isnothing(waprox)
            error("WI requires providing a valid evolver, order and w type!")
        end

        limits = Limits(cutoff=cutoff, maxdim=maxdim)
        step_method = (dt::Float64) -> ((Ψ::State) -> aprox_W(evolver, dt, Ψ; order=order, w=1, limits=limits))
        method_order = order
        return step_method, method_order

    elseif alg == "WII"
        println("No estoy seguro de que WII esté implementeado correctamente, hay que revisar")
        cutoff = get(kwargs, :cutoff, 0.) # Asigna el valor de cutoff en 0 o el valor provisto
        maxdim = get(kwargs, :maxdim, typemax(Int))
        order = get(kwargs, :order, nothing)
        evolver = get(kwargs, :evolver, nothing)
        if isnothing(evolver) | isnothing(order) | isnothing(waprox)
            error("WI requires providing a valid evolver, order and w type!")
        end

        limits = Limits(cutoff=cutoff, maxdim=maxdim)
        step_method = (dt::Float64) -> ((Ψ::State) -> aprox_W(evolver, dt, Ψ; order=order, w=2, limits=limits))
        method_order = order
        return step_method,method_order

    elseif alg == "Custom"
        custom_method::Function = get(kwargs, :custom_method, null_method)
        custom_order::Int = get(kwargs, :custom_order, 0)
        if (custom_method == null_method) | (custom_order == 0)
            error("Must specify custom method and order!")    
        end

        step_method = (dt::AbstractFloat) -> custom_method(dt; kwargs...)
        method_order = custom_order
        return step_method, method_order
    
    else
        error("Must specify a valid method!")
    end
end

"""
    step(Ψ_i::State, t::AbstractFloat, steps::Int64; step_method::Function, )

Calculates the vector obtained by repeatedly applying `step_method`
starting from `Ψ_i`, dividing a time interval of length `t` into 
`steps` parts. Return the evolved state `Ψ_f`.
"""
function step(Ψ_i::State, t::AbstractFloat, steps::Int64; step_method::Function, )
    dt = t / steps
    step_function = step_method(dt)
    Ψ_f = Ψ_i
    for _ in 1:steps
        Ψ_f = step_function(Ψ_f)
    end
    Ψ_f = State(Ψ_f, normalize(Ψ_f.state))
    return Ψ_f
end

"""
    evolution_step(
    Ψ::State, t::Float64, alg::String, steps::Int64,
    error_function::Function, tol::Float64=1e-4;kwargs...)
    alg::String="TEBD", tol::AbstractFloat=1e-4,
    maxdim::AbstractFloat=40)

Calculates the state of a system in a (pure) state `Ψ_i` after an 
evolution by `t` according to the hamiltonian `H` using the algorithm
`alg`.

`tol` and `maxdim` are used to control adaptive step sizing and truncation
error of the MPS. `steps` provides an initial guess for the number of 
steps to use during the evolution.

The function returns `Ψ_f`, the evolved State and `steps`, the number of
steps carried out.
"""
function evolution_step(
    Ψ::State, t::Float64, alg::String, steps::Int64,
    error_function::Function, tol::Float64=1e-4; kwargs...)

    accept = false # Determines if the next step of the evolution is accepted based on the error
    maxiters = 5 # Maximum number of iterations to apply in order to control the error

    # Determines which mehtod to use for evolution
    step_method, method_order = evolution_methods(alg; kwargs...)

    Ψ_f = Ψ # Creates two copies of Ψ for use in determining the error

    for iter in 0:maxiters
        Ψ_f = step(Ψ, t, steps, step_method=step_method)

        adaptive_error = get(kwargs, :adaptive_error, false)

        if !adaptive_error
            break
        end

        Ψ_2 = step(Ψ, t, steps * 2, step_method=step_method)
        
        # Se compara el error con la tolerancia, y se estima un nuevo número de pasos a usar.
        verbose = get(kwargs, :verbose, false)
        accept, steps = error_function(Ψ_f, Ψ_2, tol, steps, method_order; kwargs...)

        if accept
            break
        end

        if iter == maxiters # Condición de error para que no haga loops infinitos.
            error("Maxiters reached, cannot estimate error correctly")
        end
    end

    return Ψ_f, steps
end

"""
    experiment(
    Ψ_0::State, Tf::Float64, dt::Float64; alg::String="TEBD2", error_function::Function=default_error,
    tol::Float64=1e-4, kwargs...)

Performs an experiment in which a system in initial state `Ψ_0` evolves according to the algorithm `alg`.
The specifics of the evolution, such as the hamiltonian or evolution operators must be given in `kwargs`.

The function returns an array `Ψ_t` with the values of the state vector at times 0.0:dt:Tf

`tol` and `error_function` is used to control adaptive step sizing and truncation error of the MPS. 
"""
function experiment(
    Ψ_0::State, Tf::Float64, dt::Float64; alg::String="TEBD2", error_function::Function=default_error,
    tol::Float64=1e-4, kwargs...)

    T = 0.0:dt:Tf     # Values of T for which the state will be returned
    Ψ_t = Array{State,1}(undef, length(T)) # Array of states corresponding to the specified times
    steps::Int64 = 1

    Ψ_t[1] = Ψ_0

    if (error_function == default_error) && (alg == "TEBD2")
        error_function = TEBD2_error
    end

    verbose = get(kwargs, :verbose, false)
    if verbose
        printstyled("Comenzando evolución del sistema. \n Se obtendrá una serie de $(length(T)) estados correspondiendo a un paso temporal $(round(dt, sigdigits=2)) y un tiempo final $(round(Tf, sigdigits=2)). \n\n\n", color=:green, bold=true)
        timeini=time()
    end

    for t in eachindex(Ψ_t[2:end])

        verbose = get(kwargs, :verbose, false)
        if verbose
            printstyled("Realizando el paso número $(t) de $(length(T)), correspondiente a un tiempo $(round(T[t+1], sigdigits=2)) de $(round(Tf, sigdigits=2)). \n\n", color=:green)
            timeinistep = time()
        end

        Ψ_t[t+1], steps = evolution_step(Ψ_t[t], dt, alg, steps, error_function, tol; kwargs...)

        if verbose
            timeendstep = time()

            timestep = timeendstep - timeinistep
        
            #estimatedendtime=(timeendstep-timeini)/t*(length(T[2:end])-t) # tiempo estimado de finalización
            estimatedendtime = timestep * (length(T[2:end]) - t) # tiempo estimado de finalización
            hours = Int(estimatedendtime ÷ 3600)
            minutes = Int((estimatedendtime - 3600 * hours) ÷ 60)
            seconds = (estimatedendtime - 3600 * hours - 60 * minutes)

            printstyled("El paso temporal ha durado $(round(timestep, sigdigits=2)) s. Se estima que el proceso va a finalizar en alrededor de $(hours) horas, $(minutes) minutos y $(round(seconds, sigdigits=2)) segundos. \n\n", color=:green)
        end
    end
    
    if verbose
        endtime = time() - timeini
        hours = Int(endtime ÷ 3600)
        minutes = Int((endtime - 3600 * hours) ÷ 60)
        seconds = (endtime - 3600 * hours - 60 * minutes)
        printstyled("Ha finalizado la evolución del sistema, tardando $(hours) horas, $(minutes) minutos y $(round(seconds, sigdigits=2)) segundos en realizar la simulación. \n\n\n", color=:green, bold=true)
    end
    
    return Ψ_t
end

#--------------------------------------------------------------------------------------------------------------------------------
## Funciones para mediciones sobre series temporales de estados
#--------------------------------------------------------------------------------------------------------------------------------

function krylov_orthogonalize(statehist::Array{State})
    krylov_basis = copy(statehist)
    weights = Array{ComplexF64,2}(undef, (length(statehist), length(statehist)))
    krylov_score = Array{Float32,1}(undef, length(statehist))

    for k in eachindex(krylov_basis)
        for j in eachindex(krylov_basis[1:k-1])
            weights[j,k] = inner(krylov_basis[j], krylov_basis[k])
            krylov_basis[k] = -(krylov_basis[k], weights[j,k]*krylov_basis[j], limits=Limits(1e-8,50))
        end
        
        weights[k,k] = norm(krylov_basis[k])
        lin_dep = weights[k,k] == 0 # no se si dejarlo así o si poner una tolerancia

        krylov_score[k] += sum(Array(0:k-1).*abs.(weights[1:k,k]).^2)

        if !lin_dep
            krylov_basis[k] /= weights[k,k]
        end
    end

    return krylov_basis, weights, krylov_score
end

function obs_results(statehist, observables)
    results = Array{ComplexF64,2}(undef, (length(statehist), length(observables)))
    for k in eachindex(statehist)
        results[k,:] = last.(measure(statehist[k], observables))
    end
    return results
end
