function surrogate_fop(u)
    a = zeros(Float64, n_bilinear)
    b = zeros(Float64, n_bilinear)

    for term = 1:n_bilinear
        a[term] =
            sum(
                params[1, term, order, input] * u[input]^(order) for order = 1:n_degree-1,
                input = 1:n_input
            ) + params[1, term, n_degree, 1]
        b[term] =
            sum(
                params[2, term, order, input] * u[input]^(order) for order = 1:n_degree-1,
                input = 1:n_input
            ) + params[2, term, n_degree, 1]
    end
    t_start = time()
    fop = Model(optimizer_with_attributes(Gurobi.Optimizer, "OutputFlag" => 0))

    JuMP.@variables(
        fop,
        begin
            f[i = 1:maximum(nj), j = 1:p] >= 0, (start = rand(rng, maximum(nj), p)[i, j])
            q[1:p, 1:l] >= -1 # sometimes becomes infeasible if not allowed to take small negative values
            x[1:p, 1:r] >= -1
        end
    )

    @constraints(
        fop,
        begin
            [j = 1:p], sum(f[i, j] for i = 1:nj[j]) - sum(x[j, k] for k = 1:r) == 0
            [j = 1:p, w = 1:l],
            sum(
                a[bilinear_map[j, k, w]] * q[j, w] + b[bilinear_map[j, k, w]] * x[j, k] for
                k = 1:r
            ) - sum(lambda[i, j, w] * f[i, j] for i = 1:nj[j]) == 0
            [k = 1:r], sum(x[j, k] for j = 1:p) <= u[k] * 100
            [k = 1:r, w = 1:l],
            sum(
                a[bilinear_map[j, k, w]] * q[j, w] + b[bilinear_map[j, k, w]] * x[j, k] for
                j = 1:p
            ) - z[k, w] * sum(x[j, k] for j = 1:p) <= 0
        end
    )

    @objective(
        fop,
        Min,
        sum(1000 * c[i, j] * (f[i, j]) for j = 1:p, i = 1:maximum(nj)) +
        sum(c[i, j] * (f[i, j] - f_bar[i, j])^2 for j = 1:p, i = 1:maximum(nj)) -
        sum(d[k] * x[j, k] for j = 1:p, k = 1:r)
    )

    optimize!(fop)
    if termination_status(fop) == MOI.INFEASIBLE_OR_UNBOUNDED ||
       termination_status(fop) == MOI.INFEASIBLE ||
       termination_status(fop) == MOI.NUMERICAL_ERROR
        println("infeasible")
        return 0, zeros(Float64, maximum(nj), p), zeros(Float64, p, r)
    end
    return time() - t_start, objective_value(fop)
end
