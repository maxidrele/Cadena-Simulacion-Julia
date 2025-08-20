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
        return Diagonal([i==1 ? 1.0 : 0.0 for i in 1:n_exit])
    end
    ninv=n^-1
    prob = [(ninv+1)^-(i+1)*ninv for i in 1:n_exit]
    return Diagonal(prob)
end

#-----------------------------------------------------------------------------------------------
##  Para correr una simulación
#-----------------------------------------------------------------------------------------------

T=200.0 # Tiempo final de la simulación
dt=1.0 # Paso temporal considerado

n_oscil=7 # Número de osciladores en la cadena. Mayor o igual a 2
n_exit=5 # Número de exitaciones a considerar en los estados bosónicos

ω=5.0 # Frecuencia de oscilador armónico
U=0.0 # Anarmonía
λ=0.02 # Parámetro de interacción a primeros vecinos
n1=0.3 # Ocupación media de reservorio 1
n2=0.06 # Ocupación media de reservorio 2
γ1=0.1 # Parametro de interacción con reservorio 1
γ2=0.1 # Parametro de interacción con reservorio 2

# Algunos estados iniciales posibles.
# thermalstate, empieza en estados térmicos variando linealmente en la cadena entre n_1 y n_2 (obs que esto creo que no corresponde a fourier)
thermalstate=[thermal_matrix(n1+(n2-n1)*(j-1)/(n_oscil-1),n_exit) for j in 1:n_oscil]
# zerostate, estado en que todo está inicialmente en 0
zerostate=["0" for i in 1:n_oscil]

# Hace una corrida con los parámetros de arriba, guardando los estados en k*dt tal que 0<=k*dt<=T
hist=TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
hist

# Mide la ocupación sobre cada sitio para cada estado en la serie temporal
occu=zeros(n_oscil,length(hist))
for (k,state) in enumerate(hist)
    occu[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end

# grafica las ocupaciones en un heatmap
heatmap(1:n_oscil,0:dt:T,transpose(occu))

#-----------------------------------------------------------------------------------------------
## Algunas simulaciones para ver que pasa con diferentes parámetros de cutoff, dt, maxdim, n_exit y el estado inicial.
#-----------------------------------------------------------------------------------------------

T=20.0 # Tiempo final de la simulación
dt=1.0 # Paso temporal considerado

n_oscil=7 # Número de osciladores en la cadena. Mayor o igual a 2
n_exit=5 # Número de exitaciones a considerar en los estados bosónicos

ω=5.0 # Frecuencia de oscilador armónico
U=0.0 # Anarmonicidad
λ=0.02 # Parámetro de interacción a primeros vecinos
n1=0.3 # Ocupación media de reservorio 1
n2=0.06 # Ocupación media de reservorio 2
γ1=0.1 # Parametro de interacción con reservorio 1
γ2=0.1 # Parametro de interacción con reservorio 2

# Algunos estados iniciales posibles.
# thermalstate, empieza en estados térmicos variando linealmente en la cadena entre n_1 y n_2 (obs que esto creo que no corresponde a fourier)
thermalstate=[thermal_matrix(n1+(n2-n1)*(j-1)/(n_oscil-1),n_exit) for j in 1:n_oscil]
# zerostate, estado en que todo está inicialmente en 0
zerostate=["0" for i in 1:n_oscil]

ctf_array=[1e-7,1e-8,1e-9,1e-10]
finalstates_ctf=[TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=ctf, maxdim=40, verbose=true)[end] for ctf in ctf_array]
occu_ctf=zeros(n_oscil,length(finalstates_ctf))
for (k,state) in enumerate(finalstates_ctf)
    occu_ctf[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,ctf_array,transpose(occu_ctf))
title!("Error para t=$T")
xlabel!("Sitio")
ylabel!("Cutoff")

dt_array=[0.25,0.5,1,2]
finalstates_dt=[TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)[end] for dt in dt_array]
occu_dt=zeros(n_oscil,length(finalstates_dt))
for (k,state) in enumerate(finalstates_dt)
    occu_dt[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,dt_array,transpose(occu_dt))
title!("Error para t=$T")
xlabel!("Sitio")
ylabel!("dt")

mxd_array=[35,40,45,50]
finalstates_mxd=[TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=mxd, verbose=true)[end] for mxd in mxd_array]
occu_mxd=zeros(n_oscil,length(finalstates_mxd))
for (k,state) in enumerate(finalstates_mxd)
    occu_mxd[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,mxd_array,transpose(occu_mxd))
title!("Error para t=$T")
xlabel!("Sitio")
ylabel!("Maxdim")

nex_array=[3,5,7,9]
finalstates_nex=[TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, nex, T, dt; cutoff=1e-8, maxdim=40, verbose=true)[end] for nex in nex_array]
occu_nex=zeros(n_oscil,length(finalstates_nex))
for (k,state) in enumerate(finalstates_nex)
    occu_nex[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,nex_array,transpose(occu_nex))
title!("Error para t=$T")
xlabel!("Sitio")
ylabel!("Número de exitaciones posibles")

#= 
hist_zeros=TransportExperiment(zerostate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
# Mide la ocupación sobre cada sitio para cada estado en la serie temporal
occu_zeros=zeros(n_oscil,length(hist_zeros))
for (k,state) in enumerate(hist_zeros)
    occu_zeros[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,0:dt:T,transpose(occu_zeros))
title!("Ocupación empezando de estados vacíos")
xlabel!("Sitio")
ylabel!("Tiempo")

hist_terma=TransportExperiment(thermalstate, ω, U, λ, γ1, n1, γ2, n2, n_exit, T, dt; cutoff=1e-8, maxdim=40, verbose=true)
# Mide la ocupación sobre cada sitio para cada estado en la serie temporal
occu_terma=zeros(n_oscil,length(hist_terma))
for (k,state) in enumerate(hist_terma)
    occu_terma[:,k] .= real.(last.(measure(state,[N(i) for i in 1:n_oscil])))
end
heatmap(1:n_oscil,0:dt:T,transpose(occu_terma))
title!("Ocupación empezando de estados térmicos")
xlabel!("Sitio")
ylabel!("Tiempo")
 =#

#--------------------------------------------------------------------------------------------------------------------------------
## Análisis de datos
#--------------------------------------------------------------------------------------------------------------------------------

function discrete_aprox(tempseries, stepnum)
    len=length(tempseries)-1
    interval=len/stepnum

    aproxseries=Array{Float64,1}(undef,stepnum+1)
    for i in 0:stepnum
        t=1+i*interval
        prop=t%1
        site=Int32(floor(t))
        if prop != 0
            aproxseries[1+i]=prop*tempseries[site]+(1-prop)*tempseries[site+1]
        else
            aproxseries[1+i]=tempseries[site]
        end
    end
    return aproxseries
end