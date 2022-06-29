# import packages needed
using Catalyst, DifferentialEquations, Plots, DataFrames, GLM, Statistics, Distributions, CSV, GraphViz
using DifferentialEquations.EnsembleAnalysis

## simple reaction network model
coreMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA          #transcription and degradation of mRNA
    kₚ, mRNA --> mRNA + protein   #translation of mRNA to protein
    dₚ, protein --> ∅             #protein degradation
end ρₘ dₘ kₚ dₚ
##

## extend to sequestration model
sqstMDL = @reaction_network begin
    (ρₘ,dₘ), ∅ <--> mRNA                   #transcription and degradation of mRNA
    (ρₛ, dₛ), ∅ <--> sRNA;                  #transcription and degradation of sRNA
    (kₛₘ, rₛₘ), mRNA + sRNA <--> smRNA      #formation of sRNA-mRNA complex
    kₚ, mRNA --> mRNA + protein             #translation of mRNA to protein
    dₚ, protein --> ∅                       #protein degradation
end ρₘ dₘ ρₛ dₛ kₛₘ rₛₘ kₚ dₚ
#g = complexgraph(sqstMDL)
##

vec1 = 0.01 : 0.01 : 0.1
vec2 = 0.2 : 0.1 : 1.0
vec3 = 2.0 : 1.0 : 10.0
logRange = [vec1 ; vec2; vec3] # create a logscale range for parasweep across diff orders of magnitude

# chemical species index: 1 - mRNA, 2 - sRNA, 3 - smRNA, 4 - protein 
function singleSweep(reactionNetwork, u0, tspan, p0, pRange, trajectNum, chemSpecIndex)
    p0_copy = copy(p0) # store a copy of initial parameter vector for each iteration
    columnNames = [string.(reactionparams(reactionNetwork)) ; ["mean", "variance", "fano_factor", "tspan", "trajectories", "stationary"]]
    df = DataFrame([[] for _ = columnNames] , columnNames) # construct dataframe to hold all parameter values and statistics
    
    for j in 1 : length(string.(reactionparams(reactionNetwork))) #perform a sweep on each parameter in the network
        p0 = copy(p0_copy)
        println("Sweeping "*string.(reactionparams(reactionNetwork))[j]*"...")
        for i in pRange # for loop to iterate over parameter values
           
            p0[j] = i; # change parameter that is currently being swept over

            # Run Simulation
            @named odeSys = convert(ODESystem,reactionNetwork); # create ODE Problem
            prob = ODEProblem(odeSys,u0,tspan,p0;jac=true,sparse=true);
            ssprob = SteadyStateProblem(prob); # convert to Steady State Problem
            ss_sol = round.(solve(ssprob, SSRootfind())); # store steady state solutions
            dprob = DiscreteProblem(reactionNetwork, ss_sol, tspan, p0); # create stochastic discrete and jump problems (gillespie) with tspan input
            jprob = JumpProblem(reactionNetwork, dprob, Direct());

            rand_u0 = zeros(length(ss_sol))
            for a in 1 : length(ss_sol)
                rand_u0[a] = rand(Poisson(ss_sol[a]))
            end

            function prob_func(prob,i,repeat)
                @. prob.prob.u0 = copy(rand_u0); # draw initial conditions for discrete prob from poisson dist 
                prob
            end

            ensemble_prob = EnsembleProblem(jprob, prob_func=prob_func); # solve using ensemble problem 
            simulation = solve(ensemble_prob, SSAStepper(), EnsembleThreads(), trajectories=trajectNum);
            #display(plot(simulation, linealpha = 0.5, linewidth = 1))

            # check for stationary distribution, store results in dataframes
            push!(df, stationaryDist(simulation, chemSpecIndex, tspan, p0, trajectNum));
        end
        
        
    end
    #print(df)
    return df;
end

# chemSpecIndex corresponds to chemical species of interest: 1 - mRNA, 2 - sRNA, 3 - smRNA, 4 - protein
function stationaryDist(simulation, chemSpecIndex, tspan, pValues, trajectNum)
    ans = ["Yes", "No"];
    ts = tspan[1] : 1 : tspan[end];

    m_series,v_series = timeseries_point_meanvar(simulation,ts) # obtain mean and variance at each time point
    
    #plot(ts, m_series[chemSpecIndex,:], labels="Mean")
    #display(plot!(ts, v_series[chemSpecIndex,:], title=pValues, labels="Variance")) 

    # Linear regression to check for stationary distribution
    recent_cutoff = .20; 
    slope_threshold = 1000; # cutoff for stationary slope
    
    # obtain the x% most recent time points
    spec_mean = m_series[chemSpecIndex, :]
    spec_var = v_series[chemSpecIndex, :]
    mean_recent = spec_mean[Int(round(tspan[end] - recent_cutoff * tspan[end])):Int(tspan[end])]
    var_recent = spec_var[Int(round(tspan[end] - recent_cutoff * tspan[end])):Int(tspan[end])]
    mean_recent = mean_recent .- mean(mean_recent); # normalize data around 1
    var_recent = var_recent .- mean(var_recent);
   
    # create dataframes + conduct linear regression
    data_mSeries = DataFrame(X = 0 : 1 : length(mean_recent) - 1, Y = mean_recent)
    data_vSeries = DataFrame(X = 0 : 1 : length(var_recent) - 1, Y = var_recent)
    linMean = lm(@formula(Y ~ X), data_mSeries);
    linVar = lm(@formula(Y ~ X), data_vSeries);

    # store coefficients in tuple type? index 2 = slope
    mean_coef = coef(linMean);
    var_coef = coef(linVar);
    #println(mean_coef)
    #println(var_coef)

    if (abs(mean_coef[2]) > slope_threshold || abs(var_coef[2]) > slope_threshold) # check if distribution is stationary
        stationary = ans[2];
    else
        stationary = ans[1];
    end

    # plot linear regression for visualization
    #x = 0 : 1 : length(mean_recent) - 1
    #mean_lm(x) = mean_coef[1] + mean_coef[2] * x
    #var_lm(x) = var_coef[1] + var_coef[2] * x
    #display(plot(x, mean_lm.(x), legend=false))
    #display(plot(x, var_lm.(x), legend=false))

    # obtain mean and variance from the end of simulation
    m,v = timepoint_meanvar(simulation, ts[end])
    mean_stat = m[chemSpecIndex];
    var_stat = v[chemSpecIndex];

    # store parameter, parameter value, mean, variance, FF, stability of simulation
    return [pValues ;[mean_stat, var_stat, var_stat/mean_stat, ts[end], trajectNum, stationary]]
end


##
# perform a parameter sweep on all parameters in sequestration MDL
df = singleSweep(sqstMDL, [0.,0., 0., 0.], (0.,750.), [1.0, 0.1, 1.0, 0.1, 1.0, 0.01, 1.0, 0.01], logRange, 36, 4)
CSV.write("//Users//merylliu//Desktop//SURP 2022//noiseModel//"*"parasweep_sqst_protein.csv", df)
counter = 0;
for i in 1 : length(string.(reactionparams(sqstMDL)))
    global counter
    counter +=28
    display(plot(df[(counter-27):counter, i], df[(counter-27):counter, :fano_factor], seriestype = :scatter, title=string.(reactionparams(sqstMDL))[i]*" vs. Fano Factor, sqstMDL", xlabel=string.(reactionparams(sqstMDL))[i], ylabel="Fano-Factor",legend = false))
    
end

# compare theoretical and simulation results for transcription of sRNA, ρₛ
plot(df[57:84, :ρₛ], df[57:84, :fano_factor], seriestype = :scatter, title="Theoretcial vs. Simulation, ρₛ, sqstMDL", xlabel="ρₛ", ylabel="Fano-Factor",labels="Simulation")
ρₛ_variable = 0 : 0.01 : 10.0;
display(plot!(ρₛ_variable, 1 .+ (10) .* (1.0 ./ (1.0 .+ ρₛ_variable)), labels="Theoretical"))
##

# parameter sweep on unregulated translation coreMDL
df_2 = singleSweep(coreMDL, [0., 0.], (0., 750.), [1.0, 0.1, 1.0, 0.01], logRange, 36, 2)
CSV.write("//Users//merylliu//Desktop//SURP 2022//noiseModel//"*"parasweep_core_protein.csv", df)
counter = 0;
for i in 1 : length(string.(reactionparams(coreMDL)))
    global counter
    counter +=28
    display(plot(df_2[(counter-27):counter, i], df_2[(counter-27):counter, :fano_factor], seriestype = :scatter, title=string.(reactionparams(coreMDL))[i]*" vs. Fano Factor, coreMDL", xlabel=string.(reactionparams(sqstMDL))[i], ylabel="Fano-Factor",legend = false))
end
# compare theoretical and simulation results for burst factor vs. fano factor
plot(df_2[29:56, :dₘ], df_2[29:56, :fano_factor], seriestype = :scatter, title="Theoretical vs. Simulation, dₘ, coreMDL", xlabel="dₘ", ylabel="Fano-Factor",labels="Simulation")
dₘ_variable = 0 : 0.01 : 10.0
display(plot!(dₘ_variable, 1 .+ (1.0 ./ dₘ_variable), labels="Theoretical"))


