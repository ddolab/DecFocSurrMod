using JuMP, DataFrames, CSV, Random, Gurobi, Ipopt, JLD2
GRB_ENV = Gurobi.Env()
include("fop.jl")
include("surrogate_fop.jl")

rng = MersenneTwister(12345)
p = 4
l = 1
r = 4
nj = [2, 1, 2, 1]
c = [6 10 3 7; 16 0 13 0] / 1000
d = [9, 15, 6, 12]
f_bar = [0 0 50 350; 200 0 0 0]

lambda = zeros(Float64, maximum(nj), 4, l)
lambda[1, 1, 1] = 3
lambda[2, 1, 1] = 1
lambda[1, 2, 1] = 2
lambda[1, 3, 1] = 3.5
lambda[2, 3, 1] = 1.5
lambda[1, 4, 1] = 2.5

z = zeros(Float64, r, l)
z[1, 1] = 2.5
z[2, 1] = 1.5
z[3, 1] = 3
z[4, 1] = 2

n_samples = 1000
comp_time = zeros(Float64, n_samples, 3)
obj_val = zeros(Float64, n_samples, 3)

demand = zeros(Float64, n_samples, r)
n_bilinear = p * r * l
bilinear_map = zeros(Int, p, r, l)
count = 1
for j = 1:p
    for k = 1:r
        for w = 1:l
            bilinear_map[j, k, w] = count
            global count += 1
        end
    end
end
n_degree = 3
f = jldopen("surrogate_parameters_quad.jld2", "r")
params = f["params"]

n_input = r

global u = rand(rng, 100:200, n_samples, r) / 100
for i = 1:n_samples
    demand[i, :] .= u[i, :] * 100  # u is scaled to improve numerical performance
end

# we collect global optimal f and x values to prepare a training dataset for the E2E neural network
ff = zeros(Float64, n_samples, maximum(nj), p)
xx = zeros(Float64, n_samples, p, r)
for s = 1:n_samples
    comp_time[s, 1], obj_val[s, 1], ff[s, :, :], xx[s, :, :] = fop(demand[s, :], "global")
    comp_time[s, 2], obj_val[s, 2], = fop(demand[s, :], "local")
    comp_time[s, 3], obj_val[s, 3] = surrogate_fop(u[s, :])
end

sub_opt = zeros(Float64, n_samples, 2)
for s = 1:n_samples
    sub_opt[s, 1] = abs.(obj_val[s, 1] - obj_val[s, 2]) / abs.(obj_val[s, 1])
    sub_opt[s, 2] = abs.(obj_val[s, 1] - obj_val[s, 3]) / abs.(obj_val[s, 1])
end

df = DataFrame(
    global_true_time = comp_time[:, 1],
    local_true_time = comp_time[:, 2],
    surrogate_time = comp_time[:, 3],
)
CSV.write("computation_time_stats.csv", df)

df = DataFrame(local_true_obj = sub_opt[:, 1], surrogate_obj = sub_opt[:, 2])
CSV.write("sub_optimality_stats.csv", df)

# train_data = zeros(Float64, n_samples, r+sum(nj)+p*r)
# train_data[:, 1:r] = demand
# for s in 1:n_samples
#     counter = 1
#     for j in 1:p
#         for i in 1:nj[j]
#             train_data[s, r+counter] = ff[s, i, j]
#             counter += 1
#         end
#     end
# end

# for s in 1:n_samples
#     counter = 1
#     for j in 1:p
#         for k in 1:r
#             train_data[s, r+sum(nj)+counter] = xx[s, j, k]
#             counter += 1
#         end
#     end
# end
# df = DataFrame(train_data, :auto)
# CSV.write("E2E_nn_data.csv", df)
