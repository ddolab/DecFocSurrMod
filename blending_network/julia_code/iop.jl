function iop(u, f, x, q)
    iop = Model(
        optimizer_with_attributes(
            Ipopt.Optimizer,
            "print_level" => 5,
            "max_iter" => 10000,
            "tol" => 1e-3,
            "max_wall_time" => 3600.0,
        ),
    )
    # relaxed tolerance allows faster solution with no impact on solution quality

    @variables(
        iop,
        begin
            hat_f[s = 1:n_samples, i = 1:maximum(nj), j = 1:p] >= 0, (start = f[s, i, j])
            hat_x[s = 1:n_samples, j = 1:p, k = 1:r] >= 0, (start = x[s, j, k])
            hat_q[s = 1:n_samples, j = 1:p, w = 1:l] >= 0, (start = q[s, j, w])
            mu1[s = 1:n_samples, j = 1:p]
            mu2[s = 1:n_samples, j = 1:p, w = 1:l]
            mu3[s = 1:n_samples, k = 1:r] >= 0
            mu4[s = 1:n_samples, k = 1:r, w = 1:l] >= 0
            mu5[s = 1:n_samples, i = 1:maximum(nj), j = 1:p] >= 0
            mu6[s = 1:n_samples, j = 1:p, k = 1:r] >= 0
            a[1:n_samples, 1:n_bilinear]
            b[1:n_samples, 1:n_bilinear]
            param[i = 1:2, term = 1:n_bilinear, order = 1:n_degree, input = 1:n_input]
            abs_param[1:2, 1:n_bilinear, 1:n_degree, 1:n_input] >= 0
            slack[1:n_samples, 1:p, 1:maximum(nj)] >= 0
            slack1[1:n_samples, 1:p, 1:r] >= 0
            slack2[1:n_samples, 1:p, 1:l] >= 0
            slack3[1:n_samples, 1:p, 1:l] >= 0
            slack4[1:n_samples, 1:r, 1:l] >= 0
            slack5[1:n_samples, 1:r] >= 0
            slack6[1:n_samples, 1:r, 1:l] >= 0
            slack7[1:n_samples, 1:p, 1:maximum(nj)] >= 0
            slack8[s = 1:n_samples, j = 1:p, k = 1:r] >= 0
        end
    )

    @NLconstraints(
        iop,
        begin
            [s = 1:n_samples, j = 1:p, i = 1:nj[j]],
            1000 * c[i, j] + 2 * c[i, j] * (hat_f[s, i, j] - f_bar[i, j]) + mu1[s, j] -
            sum(mu2[s, j, w] * lambda[i, j, w] for w = 1:l) - mu5[s, i, j] <=
            slack[s, j, i]
            [s = 1:n_samples, j = 1:p, i = 1:nj[j]],
            -(
                1000 * c[i, j] + 2 * c[i, j] * (hat_f[s, i, j] - f_bar[i, j]) + mu1[s, j] -
                sum(mu2[s, j, w] * lambda[i, j, w] for w = 1:l) - mu5[s, i, j]
            ) <= slack[s, j, i]

            [s = 1:n_samples, j = 1:p, k = 1:r],
            -d[k] - mu1[s, j] +
            sum(mu2[s, j, w] * b[s, bilinear_map[j, k, w]] for w = 1:l) +
            mu3[s, k] +
            sum(mu4[s, k, w] * b[s, bilinear_map[j, k, w]] for w = 1:l) -
            sum(z[k, w] * mu4[s, k, w] for w = 1:l) - mu6[s, j, k] <= slack1[s, j, k]
            [s = 1:n_samples, j = 1:p, k = 1:r],
            -(
                -d[k] - mu1[s, j] +
                sum(mu2[s, j, w] * b[s, bilinear_map[j, k, w]] for w = 1:l) +
                mu3[s, k] +
                sum(mu4[s, k, w] * b[s, bilinear_map[j, k, w]] for w = 1:l) -
                sum(z[k, w] * mu4[s, k, w] for w = 1:l) - mu6[s, j, k]
            ) <= slack1[s, j, k]

            [s = 1:n_samples, j = 1:p, w = 1:l],
            sum(mu2[s, j, w] * a[s, bilinear_map[j, k, w]] for k = 1:r) +
            sum(mu4[s, k, w] * a[s, bilinear_map[j, k, w]] for k = 1:r) <= slack2[s, j, w]
            [s = 1:n_samples, j = 1:p, w = 1:l],
            -(
                sum(mu2[s, j, w] * a[s, bilinear_map[j, k, w]] for k = 1:r) +
                sum(mu4[s, k, w] * a[s, bilinear_map[j, k, w]] for k = 1:r)
            ) <= slack2[s, j, w]

            [s = 1:n_samples, j = 1:p],
            sum(hat_f[s, i, j] for i = 1:nj[j]) - sum(hat_x[s, j, k] for k = 1:r) == 0

            [s = 1:n_samples, j = 1:p, w = 1:l],
            sum(
                a[s, bilinear_map[j, k, w]] * hat_q[s, j, w] +
                b[s, bilinear_map[j, k, w]] * hat_x[s, j, k] for k = 1:r
            ) - sum(lambda[i, j, w] * hat_f[s, i, j] for i = 1:nj[j]) <= slack3[s, j, w]
            [s = 1:n_samples, j = 1:p, w = 1:l],
            -(
                sum(
                    a[s, bilinear_map[j, k, w]] * hat_q[s, j, w] +
                    b[s, bilinear_map[j, k, w]] * hat_x[s, j, k] for k = 1:r
                ) - sum(lambda[i, j, w] * hat_f[s, i, j] for i = 1:nj[j])
            ) <= slack3[s, j, w]

            [s = 1:n_samples, k = 1:r], sum(hat_x[s, j, k] for j = 1:p) <= demand[s, k]

            [s = 1:n_samples, k = 1:r, w = 1:l],
            sum(
                a[s, bilinear_map[j, k, w]] * hat_q[s, j, w] +
                b[s, bilinear_map[j, k, w]] * hat_x[s, j, k] for j = 1:p
            ) - z[k, w] * sum(hat_x[s, j, k] for j = 1:p) <= slack4[s, k, w]

            [s = 1:n_samples, k = 1:r],
            mu3[s, k] * (sum(hat_x[s, j, k] for j = 1:p) - demand[s, k]) <= slack5[s, k]
            [s = 1:n_samples, k = 1:r],
            -mu3[s, k] * (sum(hat_x[s, j, k] for j = 1:p) - demand[s, k]) <= slack5[s, k]

            [s = 1:n_samples, k = 1:r, w = 1:l],
            mu4[s, k, w] * (
                sum(
                    a[s, bilinear_map[j, k, w]] * hat_q[s, j, w] +
                    b[s, bilinear_map[j, k, w]] * hat_x[s, j, k] for j = 1:p
                ) - z[k, w] * sum(hat_x[s, j, k] for j = 1:p)
            ) <= slack6[s, k, w]
            [s = 1:n_samples, k = 1:r, w = 1:l],
            -mu4[s, k, w] * (
                sum(
                    a[s, bilinear_map[j, k, w]] * hat_q[s, j, w] +
                    b[s, bilinear_map[j, k, w]] * hat_x[s, j, k] for j = 1:p
                ) - z[k, w] * sum(hat_x[s, j, k] for j = 1:p)
            ) <= slack6[s, k, w]

            [s = 1:n_samples, j = 1:p, i = 1:nj[j]],
            mu5[s, i, j] * hat_f[s, i, j] <= slack7[s, j, i]
            [s = 1:n_samples, j = 1:p, i = 1:nj[j]],
            -mu5[s, i, j] * hat_f[s, i, j] <= slack7[s, j, i]

            [s = 1:n_samples, j = 1:p, k = 1:r],
            mu6[s, j, k] * hat_x[s, j, k] <= slack8[s, j, k]
            [s = 1:n_samples, j = 1:p, k = 1:r],
            -mu6[s, j, k] * hat_x[s, j, k] <= slack8[s, j, k]
        end
    )

    @constraint(
        iop,
        [s = 1:n_samples, term = 1:n_bilinear],
        a[s, term] ==
        sum(
            param[1, term, order, input] * u[s, input]^(order) for order = 1:n_degree-1,
            input = 1:n_input
        ) + param[1, term, n_degree, 1]
    )
    @constraint(
        iop,
        [s = 1:n_samples, term = 1:n_bilinear],
        b[s, term] ==
        sum(
            param[2, term, order, input] * u[s, input]^(order) for order = 1:n_degree-1,
            input = 1:n_input
        ) + param[2, term, n_degree, 1]
    )

    @constraint(
        iop,
        [i = 1:2, term = 1:n_bilinear, order = 1:n_degree, input = 1:n_input],
        abs_param[i, term, order, input] >= param[i, term, order, input]
    )
    @constraint(
        iop,
        [i = 1:2, term = 1:n_bilinear, order = 1:n_degree, input = 1:n_input],
        abs_param[i, term, order, input] >= -param[i, term, order, input]
    )

    @objective(
        iop,
        Min,
        sum((hat_f[s, i, j] - f[s, i, j])^2 for s = 1:n_samples, j = 1:p, i = 1:nj[j]) +
        sum((hat_x[s, j, k] - x[s, j, k])^2 for s = 1:n_samples, j = 1:p, k = 1:r) +
        sum((hat_q[s, j, w] - q[s, j, w])^2 for s = 1:n_samples, j = 1:p, w = 1:l) +
        1e-5 * sum(abs_param) +
        sum(slack) +
        sum(slack1) +
        sum(slack2) +
        sum(slack3) +
        sum(slack4) +
        sum(slack5) +
        sum(slack6) +
        sum(slack7) +
        sum(slack8)
    )

    optimize!(iop)

    return value.(param)
end
