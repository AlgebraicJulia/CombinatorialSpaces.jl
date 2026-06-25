# kelvin_helmholtz_mpi/kh_output.jl
using HDF5
using CairoMakie

const cart_comm = topo.cart_comm
const IMGDIR = joinpath(@__DIR__, "imgs")

# ── DataHandler ───────────────────────────────────────────────────────────────
const io_stream = DataStream("kh_2D", OUTFILE, Datum[Datum{Quad,2}("rho", "fields", FT), Datum{Quad,2}("Theta", "fields", FT)])
const handler = DataHandler(io_stream, topo)

output_leader(topo) && rm(OUTPUT_DIR; recursive = true, force = true)
MPI.Barrier(cart_comm)
output_leader(topo) && mkpath(OUTPUT_DIR)
MPI.Barrier(cart_comm)

create_hdf5!(handler, m_dims)
MPI.Barrier(cart_comm)

# ── Busy-wait loop ────────────────────────────────────────────────────────────
while true
    tag = output_from_worker(topo)
    tag == SIGNAL_DONE && break
    tag == SIGNAL_WRITE && write_output!(handler)
end
MPI.Barrier(cart_comm)

# # ── Post-processing (output leader only) ──────────────────────────────────────
# if output_leader(topo)
#     mkpath(IMGDIR)

#     h5open(OUTFILE, "r") do h5
#         for field in ("rho", "Theta")
#             dset = h5["fields/$field"]
#             extent = HDF5.get_extent_dims(HDF5.dataspace(dset))[1]
#             ntimes = extent[1]

#             ic = dset[1, :, :]
#             final = dset[ntimes, :, :]
#             cr = (minimum(ic), maximum(ic))

#             for (data, label) in ((ic, "IC"), (final, "final"))
#                 fig = Figure(; size = (700, 600))
#                 ax = Axis(fig[1, 1]; title = "$field | $label", xlabel = "x", ylabel = "y")
#                 hm = heatmap!(ax, data; colorrange = cr)
#                 Colorbar(fig[1, 2], hm)
#                 save(joinpath(IMGDIR, "$(field)_$(lowercase(label)).png"), fig)
#             end

#             let frame_obs = Observable(dset[1, :, :])
#                 fig = Figure(; size = (700, 600))
#                 ax = Axis(fig[1, 1]; xlabel = "x", ylabel = "y")
#                 hm = heatmap!(ax, frame_obs; colorrange = cr)
#                 Colorbar(fig[1, 2], hm)
#                 record(fig, joinpath(IMGDIR, "$(field).gif"), 1:ntimes; framerate = 15) do i
#                     frame_obs[] = dset[i, :, :]
#                     return ax.title[] = "$field | t = $(round((i-1)*SAVETIME, digits=3))"
#                 end
#             end
#         end
#         return println("Output leader: plots and GIFs saved to $IMGDIR")
#     end
# end