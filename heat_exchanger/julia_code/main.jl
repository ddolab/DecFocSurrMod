using JuMP, Gurobi, Ipopt, LinearAlgebra, Random, PyPlot, DataFrames, CSV
using JLD2

include("fop.jl")
include("init_params.jl")
include("data_structs.jl")
include("pen_refo.jl")

rng = MersenneTwister(12345)
gurobi_env = Gurobi.Env()
instances = 10
samples = [5, 10, 25, 50, 75, 100, 200]
degree = 4
prediction_error = zeros(Float64, instances, length(samples))
instance = 1
while instance <= instances
    global flag = 0
    test_u = rand(rng, 0:200, 100) / 10 .+ 573
    global u = rand(rng, 0:200, 200) / 10 .+ 573
    global x = [forward_problem(u[i]) for i = 1:200]
    true_x = [forward_problem(test_u[i]) for i = 1:100]
    column = 1
    for I in samples
        global NoS = I
        cons_para = init_constraint_params()
        obj_para, dual_var = init_objective_params(cons_para)
        s = [ones(Float64, 2, I), ones(Float64, 5, I), ones(Float64, I)]
        global vars = IOPVars(deepcopy(x), dual_var, s, obj_para, cons_para)

        penalty_coeffs = deepcopy(s)
        penalty_coeffs = 1 .* penalty_coeffs
        global params = AlgParams(10, 1e6, penalty_coeffs, 1000, 5)
        inverse_problem()
        if flag == 1
            break
        end
        jldsave(
            "params_$(instance)_I_$(I).jld2";
            obj_params = vars.obj_params,
            cons_params = vars.constraint_params,
        )
        pred_x = [
            surrogate_fop(vars.constraint_params, vars.obj_params, test_u[i]) for i = 1:100
        ]
        df = DataFrame(
            true_Qc = [true_x[i][1] for i = 1:length(true_x)],
            true_FH = [true_x[i][2] for i = 1:length(true_x)],
            pred_Qc = [pred_x[i][1] for i = 1:length(pred_x)],
            pred_FH = [pred_x[i][2] for i = 1:length(pred_x)],
        )
        pred_error = norm(df[!, "true_Qc"] - df[!, "pred_Qc"], 1)
        prediction_error[instance, column] = pred_error
        column = column + 1
        CSV.write("results_instance_test_$(instance)_I_$(I)_test.csv", df)
        figure()
        scatter(test_u, [pred_x[i][1] for i = 1:100], color = "blue", label = "predicted")
        scatter(test_u, [true_x[i][1] for i = 1:100], color = "red", label = "true")
        savefig("validation_data_test_$(instance)_I_$(I)_test.png")

        Temp_range = 573:0.2:593 # Finer the grid, better the feasible region plot. 
        min = zeros(Float64, length(Temp_range))
        max = zeros(Float64, length(Temp_range))
        i = 0
        for T5 in Temp_range
            i += 1
            min[i], max[i] =
                SurrogateBoundingProblem(vars.obj_params, vars.constraint_params, T5)
        end
        PyPlot.figure()
        PyPlot.fill_between(Temp_range, max, min, facecolor = "blue", alpha = 0.2)
        PyPlot.xlim([minimum(Temp_range), maximum(Temp_range)])
        PyPlot.ylim([0, 700])
        labels = minimum(Temp_range):4:maximum(Temp_range)
        PyPlot.xticks(labels, string.(labels))
        PyPlot.savefig(
            "feasible_region_instance_$(instance)_I_$(I).png",
            bbox_inches = "tight",
        )
        PyPlot.close("all")
    end
    if flag == 0
        global instance += 1
    end
end
temp_df = DataFrame(prediction_error, :auto)
CSV.write("full_pred_error.csv", temp_df)
