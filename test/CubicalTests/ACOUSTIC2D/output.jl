# kelvin_helmholtz_mpi/kh_output.jl
using HDF5
using CairoMakie

const cart_comm = topo.cart_comm

# ── DataHandler ───────────────────────────────────────────────────────────────
const io_stream = DataStream("kh_2D", OUTFILE, Datum[
    Datum{Quad,2}("rho",   "fields", FT),
    Datum{Quad,2}("Theta", "fields", FT),
    Datum{Edge,2}("U",     "fields", FT),
])
const handler = DataHandler(io_stream, topo)

create_hdf5!(handler, m_dims)
MPI.Barrier(cart_comm)

# ── Busy-wait loop ────────────────────────────────────────────────────────────
while true
    tag = output_from_worker(topo)
    tag == SIGNAL_DONE && break
    tag == SIGNAL_WRITE && write_output!(handler)
end
MPI.Barrier(cart_comm)

# ── Post-processing (output leader only) ──────────────────────────────────────
if output_leader(topo)
    # Build a global mesh for DEC operators (no halo needed for post-processing)
    NXQ, NYQ = m_dims[1] - 1, m_dims[2] - 1
    s_global = UniformCubicalComplex2D(m_dims[1], m_dims[2], FT(LX), FT(LY))

    h5open(OUTFILE, "r") do h5

        # ── scalar fields: rho and Theta ──────────────────────────────────────
        for (field, color) in (("rho", Reverse(:oslo)), ("Theta", :magma))
            dset  = h5["fields/$field"]
            ntimes = HDF5.get_extent_dims(HDF5.dataspace(dset))[1][1]

            ic    = dset[1,       :, :]
            final = dset[ntimes,  :, :]

            # cr_min, cr_max = minimum(ic), maximum(ic)
            # cr = cr_min == cr_max ? (cr_min - FT(1), cr_max + FT(1)) : (cr_min, cr_max)

            for (data, label) in ((ic, "IC"), (final, "final"))
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; title = "$field | $label", xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, data; colormap = color)
                Colorbar(fig[1, 2], hm)
                save(joinpath(IMGDIR, "$(field)_$(lowercase(label)).png"), fig)
            end

            let frame_obs = Observable(dset[1, :, :])
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, frame_obs; colormap = color)
                Colorbar(fig[1, 2], hm)
                record(fig, joinpath(IMGDIR, "$(field).mp4"), 1:ntimes; framerate = 15) do i
                    frame_obs[] = dset[i, :, :]
                    ax.title[]  = "$field | t = $(round((i-1)*SAVETIME, digits=3))"
                end
            end
        end

        # ── momentum components via sharp_dd ──────────────────────────────────
        # U is stored as two edge families: U_1 (x-edges) and U_2 (y-edges).
        # Concatenate them to form the full dual 1-form vector for sharp_dd.
        dset_ux = h5["fields/U_1"]   # shape (ntimes, nxe(s), ny(s))
        dset_uy = h5["fields/U_2"]   # shape (ntimes, nx(s), nye(s))
        ntimes  = HDF5.get_extent_dims(HDF5.dataspace(dset_ux))[1][1]

        # Helper: reconstruct (X_quad, Y_quad) arrays at a given timestep
        function momentum_xy(t::Int)
            ux_mat = dset_ux[t, :, :]   # (nxe, ny)
            uy_mat = dset_uy[t, :, :]   # (nx,  nye)
            # Flatten in the same order as coord_to_edge: x-family then y-family
            u_flat = vcat(vec(ux_mat), vec(uy_mat))
            X, Y   = sharp_dd(s_global, u_flat)
            # Reshape to (nxq, nyq) for heatmap (quads laid out row-major)
            X_mat  = reshape(X, NXQ, NYQ)
            Y_mat  = reshape(Y, NXQ, NYQ)
            return X_mat, Y_mat
        end

        ic_X,    ic_Y    = momentum_xy(1)
        final_X, final_Y = momentum_xy(ntimes)

        for (comp_label, ic_data, final_data) in (
                ("momentum_X", ic_X, final_X),
                ("momentum_Y", ic_Y, final_Y))

            cr_min, cr_max = minimum(ic_data), maximum(ic_data)
            cr = cr_min == cr_max ? (cr_min - FT(1), cr_max + FT(1)) : (cr_min, cr_max)

            for (data, label) in ((ic_data, "IC"), (final_data, "final"))
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; title = "$comp_label | $label", xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, data; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                save(joinpath(IMGDIR, "$(comp_label)_$(lowercase(label)).png"), fig)
            end

            let frame_obs = Observable(momentum_xy(1)[comp_label == "momentum_X" ? 1 : 2])
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, frame_obs; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                record(fig, joinpath(IMGDIR, "$(comp_label).mp4"), 1:ntimes; framerate = 15) do i
                    Xm, Ym      = momentum_xy(i)
                    frame_obs[] = comp_label == "momentum_X" ? Xm : Ym
                    ax.title[]  = "$comp_label | t = $(round((i-1)*SAVETIME, digits=3))"
                end
            end
        end

        println("Output leader: plots and videos saved to $IMGDIR")
    end
end