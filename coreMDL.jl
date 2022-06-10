using Catalyst, DifferentialEquations, Plots
using DifferentialEquations.EnsembleAnalysis
Threads.nthreads()

coreMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA          #transcription and degradation of mRNA
    kₚ, mRNA --> mRNA + protein   #translation of mRNA to protein
    dₚ, protein --> ∅             #protein degradation
  end ρₘ dₘ kₚ dₚ
p = [1.0, 0.1, 1.0, 0.01];        #here we define the vector of parameter values ordered as follows: [ρₘ, ρₛ, kₚ, dₚ] - note: you can also use other types to pass the parameters, e.g. Tuple types: p = (1.0, 0.1, 1.0, 0.01)
tspan = (0.,500.);                #here we define the timespan, we don't need to define this here but will need this later
u0 = [0.,0.];                     #here we define the initial condition for our two variables [mRNA, protein]

# solve ODEs
oprob = ODEProblem(coreMDL, u0, tspan, p);
osol  = solve(oprob, Tsit5());
plot(osol)

# solve JumpProblem
dprob = DiscreteProblem(coreMDL, u0, tspan, p);
jprob = JumpProblem(coreMDL, dprob, Direct());
jsol = solve(jprob, SSAStepper());
plot!(jsol) 

ensemble_prob = EnsembleProblem(jprob)
sim = solve(ensemble_prob,SSAStepper(),EnsembleThreads(),trajectories=18)

plot(sim, vars=(0,2), inealpha=0.4)

@named odesys = convert(ODESystem,coreMDL);
prob = ODEProblem(odesys,u0,tspan,p;jac=true,sparse=true)
ssprob = SteadyStateProblem(prob)
ss_sol = round.(solve(ssprob,SSRootfind()))

## solve JumpProblem
tspan2 = (0.,200.)
dprob2 = DiscreteProblem(coreMDL, ss_sol, tspan2, p);
jprob2 = JumpProblem(coreMDL, dprob2, Direct());
jsol2 = solve(jprob2, SSAStepper());
plot(jsol2)  

ensemble_prob2 = EnsembleProblem(jprob2)
sim2 = solve(ensemble_prob2,SSAStepper(),EnsembleThreads(),trajectories=16)


plot(sim2, vars=(0,2), inealpha=0.4)

