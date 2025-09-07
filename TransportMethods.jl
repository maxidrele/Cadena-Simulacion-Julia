workdir=pwd()
include(workdir*"/ChainSims.jl")
import TensorMixedStates: dag, Dissipator, Id, Mixed, Pure, PreMPO, State, System, exp, ⊗
import Combinatorics: combinations
import LinearAlgebra: ishermitian, eigen, Diagonal, hermitianpart, eigvecs, diag, dot, tr
import ITensors: ITensor, state, inds, scalar
import Tables: matrix, table
import CSV: write, File

using Plots
using TensorMixedStates
using .Bosons: Boson, N, A
using DataFrames
using CSV
#λ es U03BB λ
#⊗ es U2297 ⊗
#÷ es U00F7 ÷

"""
TODO:   Agregar conservación de quantum numbers al estado.
        Agregar función que estime el adaptive time steps en base al valor calculado para un operador.
        
"""

#--------------------------------------------------------------------------------------------------------------------------------
## Definición de constantes
#--------------------------------------------------------------------------------------------------------------------------------

At=dag(A)

function make_state(data,maxn::Int)
    chain = System(length(data),Boson(maxn+1))
    vector_state = State(Mixed(),chain,data)
end

function make_state(data,system::System)
    vector_state = State(Mixed(),system,data)
end

function H0(ω,U)
    H=ω*(N)+U*N*(Id-N)/2
end

function Hint(λ)
    H=λ*((A⊗At)+(At⊗A))
end

# Funciones que crean operadores de evolución unitarios para diferentes sitios

function Hloc(i,len,ω,U,λ)
    if i==1
        return (H0(ω,0)⊗Id+Hint(λ))(i,i+1)
    elseif i==len
        return H0(ω,0)(i)
    else
        return (H0(ω,U)⊗Id+Hint(λ))(i,i+1)
    end
end

function Uloc(dt,i,len,ω,U,λ)
    if i==len
        return exp(-im*H0(ω,0)*dt)(i)  # borde: resonador
    elseif i==1
        return exp(-im*(H0(ω,0)⊗Id+Hint(λ))*dt)(i,i+1)  # borde izquierdo
    else
        return exp(-im*(H0(ω,U)⊗Id+Hint(λ))*dt)(i,i+1)  # transmon en medio
    end
end

# hamiltoniano total del sistema
function Hamiltonian(len,ω,U,λ)
    Hamiltonian = sum([Hloc(i,len,ω,U,λ) for i in 1:len])
end

function Thermal(γ, n)
    D = γ * ((1+n)*Dissipator(A)+n*Dissipator(At))
end

# El hamiltoniano del sistema se puede descomponer en 2 sumas de hamiltonianos locales en sitios pares e impares, en donde los términos de cada suma conmutan entre si.
# Funcion que genera las 2 funciones que dan la evolución en un dt correspondientes a las partes internamente conmutantes del hamiltoniano
# se espera que sitetype sea 1,2.
function conmuting_part(sitetype,len,ω,U,λ)
    return dt -> prod([Uloc(dt,i,len,ω,U,λ) for i in 1:len if i%2==sitetype%2])
end

# Lista con las funciones correspondientes a cada partición del hamiltoniano.
function Ufuncs(len,ω,U,λ)
    Ufuncs=[conmuting_part(i,len,ω,U,λ) for i in 1:2]
end

function TransportExperiment(initial_state, ω, U, λ, γ1, n1, γ2, n2, maxn, T, dt; cutoff=1e-8, maxdim=40, verbose=false, adaptive_error=false, tol=1e-4)
    len=length(initial_state)
    vector_state=make_state(initial_state, maxn)
    H=Hamiltonian(len,ω,U,λ)
    D1=Thermal(γ1, n1)(1)
    D2=Thermal(γ2, n2)(len)

    evolver = -im*H+D1+D2
    evolver=PreMPO(vector_state,evolver)
    return experiment(vector_state,T,dt,alg="TDVP",evolver=evolver,cutoff=cutoff,maxdim=maxdim,adaptive_error=adaptive_error,tol=tol,verbose=verbose)
end

function thermal_matrix(n,n_exit)
    if n==0
        return Matrix(Diagonal([i==1 ? 1.0 : 0.0 for i in 1:n_exit+1]))
    end
    ninv=n^-1
    prob = [(ninv+1)^-(i+1)*ninv for i in 0:n_exit]
    return Matrix(Diagonal(prob))
end

#-----------------------------------------------------------------------------------------------
##  Para correr una simulación
#-----------------------------------------------------------------------------------------------

#Parametros

T=100.0 # Tiempo final de la simulación
dt=10.0 # Paso temporal considerado

n_oscil=4 # Número de osciladores en la cadena. Mayor o igual a 2
n_exit=4 # Número de exitaciones a considerar en los estados bosónicos

ω=5.0 # Frecuencia de oscilador armónico
ω_h = 3.0 # Frecuencia del reservorio 1
ω_c = 2.0 # Frecuencia del reservorio 2
U=0.05 # Anarmonía
λ=0.02 # Parámetro de interacción a primeros vecinos
n_h = 1/(exp(ω/ω_h)-1) # Ocupación media de reservorio 1
n_c = 1/(exp(ω/ω_c)-1) # Ocupación media de reservorio 2
γ1=0.5*(2*pi)^-1 # Parametro de interacción con reservorio 1
γ2=0.5*(2*pi)^-1 # Parametro de interacción con reservorio 2

e1 = ((4*λ^2)*(γ1-γ2) + γ1*γ2^2 + γ2*γ1^2)/((4*λ^2 + γ1*γ2)*(γ1+γ2))
et = ((4*λ^2)*(γ1-γ2) + γ1*γ2^2 - γ2*γ1^2)/((4*λ^2 + γ1*γ2)*(γ1+γ2))
eN = ((4*λ^2)*(γ1-γ2) - γ1*γ2^2 - γ2*γ1^2)/((4*λ^2 + γ1*γ2)*(γ1+γ2))

n_1 = (n_h+n_c)/2 + (n_h-n_c)*e1/2
n_t = (n_h+n_c)/2 + (n_h-n_c)*et/2
n_N = (n_h+n_c)/2 + (n_h-n_c)*eN/2
# Algunos estados iniciales posibles.
# thermalstate, empieza en estados térmicos variando linealmente en la cadena entre n_1 y n_2 (obs que esto creo que no corresponde a fourier)
thermalstate=[
    j == 1 ? thermal_matrix(n_1,n_exit) :
    j == n_oscil ? thermal_matrix(n_N,n_exit) : 
                    thermal_matrix(n_t,n_exit) 
    for j in 1:n_oscil]
# zerostate, estado en que todo está inicialmente en 0


## Hace una corrida con los parámetros de arriba, guardando los estados en k*dt tal que 0<=k*dt<=T

hist_termal=TransportExperiment(thermalstate, ω, U, λ, γ1, n_h, γ2, n_c, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
hist_termal


# Mide la ocupación sobre cada sitio para cada estado en la serie temporal
occu=zeros(n_oscil,length(hist_termal))
for (k,state) in enumerate(hist_termal)
    occu[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end

# Vector de tiempos
times = collect(0:dt:T)

# Pasar a formato largo
rows = []
for (k, t) in enumerate(times)
    for i in 1:n_oscil
        push!(rows, (t, i, occu[i,k]))
    end
end

df = DataFrame(rows, [:tiempo, :oscilador, :ocupacion])

## Guardar en CSV
CSV.write("ocupaciones.csv", df)
