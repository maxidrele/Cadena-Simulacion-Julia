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
    H=ω*(N+Id/2)+U*N*(Id-N)
end

function Hint(λ)
    H=λ*((A⊗At)+(At⊗A))
end

# Funciones que crean operadores de evolución unitarios para diferentes sitios

function Hloc(i,len,ω,U,λ)
    if i==len
        return H0(ω,U)(i)
    else
        return (H0(ω,U)⊗Id+Hint(λ))(i,i+1)
    end
end

function Uloc(dt,i,len,ω,U,λ)
    if i==len
        return exp(-im*H0(ω,U)*dt)(i)
    else
        return exp(-im*(H0(ω,U)⊗Id+Hint(λ))*dt)(i,i+1)
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
        return Diagonal([i==1 ? 1.0 : 0.0 for i in 0:n_exit])
    end
    
    prob = [n^i / ((n+1)^(i+1)) for i in 0:n_exit]
    return Matrix(Diagonal(prob))
end

#-----------------------------------------------------------------------------------------------
##  Para correr una simulación
#-----------------------------------------------------------------------------------------------


n_oscil=4 # Número de osciladores en la cadena. Mayor o igual a 2
n_exit=4 # Número de exitaciones a considerar en los estados bosónicos

ω=5.0 # Frecuencia de oscilador armónico
U=0.2 # Anarmonía
λ=0.005 # Parámetro de interacción a primeros vecinos
n1=1.15 # Ocupación media de reservorio 1
n2=0.09 # Ocupación media de reservorio 2
γ1=0.5 # Parametro de interacción con reservorio 1
γ2=0.1 # Parametro de interacción con reservorio 2

T=150/(1/γ1) # Tiempo final de la simulación
dt=1.0 # Paso temporal considerado


n_prom = (n1 + n2)/2
delta_n = (n1 - n2)/2

e_1 = (4*λ^2*(γ1-γ2) + γ1*γ2^2 + (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

e_n = (4*λ^2*(γ1-γ2) + γ1*γ2^2 - (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

e_N = (4*λ^2*(γ1-γ2) - γ1*γ2^2 - (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

n_term1 = n_prom + e_1*delta_n
n_termN = n_prom + e_N*delta_n

# Algunos estados iniciales posibles.
# thermalstate, empieza en estados térmicos variando linealmente en la cadena entre n_1 y n_2 (obs que esto creo que no corresponde a fourier)
thermalstate=[thermal_matrix(n_term1+(n_termN-n_term1)*(j-1)/(n_oscil-1),n_exit) for j in 1:n_oscil]
# zerostate, estado en que todo está inicialmente en 0
#zerostate=["0" for i in 1:n_oscil]

# Hace una corrida con los parámetros de arriba, guardando los estados en k*dt tal que 0<=k*dt<=T
hist=TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
hist

hist_term=hist=TransportExperiment(thermalstate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
hist_term

# Mide la ocupación sobre cada sitio para cada estado en la serie temporal
occu=zeros(n_oscil,length(hist))
for (k,state) in enumerate(hist)
    occu[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end


# Crear vector de tiempos
times = 0:dt:T

# valores auxiliares


e_1 = (4*λ^2*(γ1-γ2) + γ1*γ2^2 + (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

e_n = (4*λ^2*(γ1-γ2) + γ1*γ2^2 - (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

e_N = (4*λ^2*(γ1-γ2) - γ1*γ2^2 - (γ1^2)*γ2) /
      ((4*λ^2 + γ1*γ2)*(γ1 + γ2))

# Gráfico con todas las ocupaciones
plot(times, transpose(occu), 
     #label=["Sitio 1" "Sitio 2" "Sitio 3" "Sitio 4"],
     xlabel="Tiempo",
     ylabel="Número de ocupación",
     title="Evolución temporal de la ocupación",
     linewidth=2,)
     #legend=:right)
hline!([n_prom + e_1*delta_n], color=:red,   linestyle=:dot, label="")
hline!([n_prom + e_n*delta_n], color=:black, linestyle=:dot, label="")
hline!([n_prom + e_N*delta_n], color=:blue,  linestyle=:dot, label="")
savefig("ocupaciones.png")
# grafica las ocupaciones en un heatmap


#-----------------------------------------------------------------------------------------------
## Algunas simulaciones para ver que pasa con diferentes parámetros de cutoff, dt, maxdim, n_exit y el estado inicial.
#-----------------------------------------------------------------------------------------------

