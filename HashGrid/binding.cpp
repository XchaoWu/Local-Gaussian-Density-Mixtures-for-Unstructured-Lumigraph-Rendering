#include <pybind11/pybind11.h>
#include "hashgrid.h"
#include "sampler.h"
#include "rendering.h"

namespace py = pybind11;


PYBIND11_MODULE(HASHGRID, m){

    m.doc()="Pytorch version of hash grid";

    m.def("embedding_forward_cuda", &embedding_forward_cuda, "");

    m.def("embedding_backward_cuda", &embedding_backward_cuda, "");


    m.def("embedding_bg_forward_cuda", &embedding_bg_forward_cuda, "");

    m.def("embedding_bg_backward_cuda", &embedding_bg_backward_cuda, "");

    m.def("encoding_forward_cuda", &encoding_forward_cuda, "");

    m.def("encoding_backward_cuda", &encoding_backward_cuda, "");
    
    m.def("rendering_cuda", &rendering_cuda, "");

    m.def("ray_block_intersection", &ray_block_intersection, "");
    m.def("sample_points", &sample_points, "");
    m.def("prepare_points", &prepare_points, "");
    m.def("sort_by_key", &sort_by_key, "");
    m.def("pts_inference", &pts_inference, "");
    m.def("accumulate_color", &accumulate_color, "");
    m.def("ray_firsthit_block", &ray_firsthit_block, "");
    m.def("inverse_z_sampling", &inverse_z_sampling, "");
    m.def("bg_pts_inference", &bg_pts_inference, "");
    m.def("bg_pts_inference_v2", &bg_pts_inference_v2, "");
    m.def("get_last_block", &get_last_block, "");
    m.def("update_outgoing_bidx", &update_outgoing_bidx, "");
    m.def("update_outgoing_bidx_v2", &update_outgoing_bidx_v2, "");
    m.def("process_occupied_grid", &process_occupied_grid, "");

    py::class_<Sampler>(m, "Sampler")
        .def(py::init<>())
        .def("build", &Sampler::build)
        .def("rebuild", &Sampler::rebuild)
        .def("samplePoints", &Sampler::samplePoints);

}