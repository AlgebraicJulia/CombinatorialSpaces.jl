# kelvin_helmholtz_3d_mpi/output.jl
using HDF5
using CairoMakie

const cart_comm = topo.cart_comm

# ── DataHandler ───────────────────────────────────────────────────────────────
const io_stream = DataStream("acoustic_3D", OUTFILE, Datum[
    Datum{Boid,3}("rho",   "fields", FT),
    Datum{Boid,3}("Theta", "fields", FT),
    Datum{Quad,3}("U",     "fields", FT),
])
const handler = DataHandler(io_stream, topo)

create_hdf5!(handler, m_dims)
MPI.Barrier(cart_comm)

# ── Busy-wait loop ────────────────────────────────────────────────────────────
while true
    tag = output_from_worker(topo)
    tag == SIGNAL_DONE  && break
    tag == SIGNAL_WRITE && write_output!(handler)
end
MPI.Barrier(cart_comm)

# ── Post-processing (output leader only) ──────────────────────────────────────
if output_leader(topo)
    # Global mesh for DEC post-processing (no halo needed)
    NXB = m_dims[1] - 1
    NYB = m_dims[2] - 1
    NZB = m_dims[3] - 1
    s_global = UniformCubicalComplex3D(m_dims[1], m_dims[2], m_dims[3], FT(LX), FT(LY), FT(LZ))

    # Mid-plane z-index for 2D slice visualisations
    z_mid = div(NZB, 2) + 1

    h5open(OUTFILE, "r") do h5

        # ── scalar fields: rho and Theta (boids stored flat) ──────────────────
        for field in ("rho", "Theta")
            dset   = h5["fields/$field"]
            ntimes = HDF5.get_extent_dims(HDF5.dataspace(dset))[1][1]

            # Reshape flat boid vector → (NXB, NYB, NZB) at each timestep
            ic_flat    = dset[1,      :]
            final_flat = dset[ntimes, :]
            ic    = reshape(ic_flat,    NXB, NYB, NZB)[:, :, z_mid]
            final = reshape(final_flat, NXB, NYB, NZB)[:, :, z_mid]

            for (data, label) in ((ic, "IC"), (final, "final"))
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; title = "$field z-slice | $label", xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, data)
                Colorbar(fig[1, 2], hm)
                save(joinpath(IMGDIR, "$(field)_$(lowercase(label)).png"), fig)
            end

            let frame_obs = Observable(reshape(dset[1, :], NXB, NYB, NZB)[:, :, z_mid])
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, frame_obs)
                Colorbar(fig[1, 2], hm)
                record(fig, joinpath(IMGDIR, "$(field).mp4"), 1:ntimes; framerate = 15) do i
                    frame_obs[] = reshape(dset[i, :], NXB, NYB, NZB)[:, :, z_mid]
                    ax.title[]  = "$field z-slice | t = $(round((i-1)*SAVETIME, digits=3))"
                end
            end
        end

        # ── momentum components via sharp_dd ──────────────────────────────────
        # U is stored as three edge families: U_1 (x-edges), U_2 (y-edges), U_3 (z-edges).
        dset_ux = h5["fields/U_1"]
        dset_uy = h5["fields/U_2"]
        dset_uz = h5["fields/U_3"]
        ntimes  = HDF5.get_extent_dims(HDF5.dataspace(dset_ux))[1][1]

        # Reconstruct sharp_dd vector components at a given timestep.
        # Returns (X, Y, Z) arrays shaped (NXB, NYB, NZB).
        function momentum_xyz(t::Int)
            ux_flat = dset_ux[t, :]
            uy_flat = dset_uy[t, :]
            uz_flat = dset_uz[t, :]
            u_flat  = vcat(ux_flat, uy_flat, uz_flat)
            X, Y, Z = sharp_dd(s_global, u_flat)
            return (reshape(X, NXB, NYB, NZB),
                    reshape(Y, NXB, NYB, NZB),
                    reshape(Z, NXB, NYB, NZB))
        end

        ic_X,    ic_Y,    ic_Z    = momentum_xyz(1)
        final_X, final_Y, final_Z = momentum_xyz(ntimes)

        for (comp_label, ic_data, final_data) in (
                ("momentum_X", ic_X[:, :, z_mid], final_X[:, :, z_mid]),
                ("momentum_Y", ic_Y[:, :, z_mid], final_Y[:, :, z_mid]),
                ("momentum_Z", ic_Z[:, :, z_mid], final_Z[:, :, z_mid]))

            cr_min, cr_max = minimum(ic_data), maximum(ic_data)
            cr = cr_min == cr_max ? (cr_min - FT(1), cr_max + FT(1)) : (cr_min, cr_max)

            for (data, label) in ((ic_data, "IC"), (final_data, "final"))
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; title = "$comp_label z-slice | $label", xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, data; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                save(joinpath(IMGDIR, "$(comp_label)_$(lowercase(label)).png"), fig)
            end

            let frame_obs = Observable(momentum_xyz(1)[comp_label == "momentum_X" ? 1 :
                                                        comp_label == "momentum_Y" ? 2 : 3][:, :, z_mid])
                fig = Figure(; size = (700, 600))
                ax  = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
                hm  = heatmap!(ax, frame_obs; colorrange = cr)
                Colorbar(fig[1, 2], hm)
                record(fig, joinpath(IMGDIR, "$(comp_label).mp4"), 1:ntimes; framerate = 15) do i
                    comp_idx = comp_label == "momentum_X" ? 1 : comp_label == "momentum_Y" ? 2 : 3
                    frame_obs[] = momentum_xyz(i)[comp_idx][:, :, z_mid]
                    ax.title[]  = "$comp_label z-slice | t = $(round((i-1)*SAVETIME, digits=3))"
                end
            end
        end

        println("Output leader: plots and videos saved to $IMGDIR")
    end
end