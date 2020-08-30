//
// Created by 朱世杨 on 2020/8/20.
//

#include "hysample.h"
#include "utils.h"

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
           torch::Tensor row_ids, torch::Tensor col_ids, int64_t num_neighbors) {

    CHECK_CPU(rowptr);
    CHECK_CPU(col);
    CHECK_CPU(row_ids);
    CHECK_CPU(col_ids);
    CHECK_INPUT(row_ids.dim() == 1);

    auto rowptr_data = rowptr.data_ptr<int64_t>();
    auto col_data = col.data_ptr<int64_t>();
    auto rowcount_data = rowcount.data_ptr<int64_t>();
    auto row_ids_data = row_ids.data_ptr<int64_t>();
    auto col_ids_data = col_ids.data_ptr<int64_t>();

    auto out_rowptr = torch::empty(row_ids.size(0) + 1, rowptr.options());
    auto out_rowptr_data = out_rowptr.data_ptr<int64_t>();
    out_rowptr_data[0] = 0;

    std::vector <int64_t> cols;
    std::vector <int64_t> new_col_ids;
    std::unordered_map <int64_t, int64_t> col_id2idx_map;

    for (int64_t n = 0; n < col_ids.size(0); n++) {
        int64_t i = col_ids_data[n];
        col_id2idx_map[i] = n;
        new_col_ids.push_back(i);
    }

    if (num_neighbors < 0) { // No sampling ======================================

        int64_t row_id, col_id, col_idx, offset = 0;
        for (int64_t i = 0; i < row_ids.size(0); i++) {
            row_id = row_ids_data[i];

            for (int64_t j = 0; j < rowcount_data[row_id]; j++) {
                col_idx = rowptr_data[row_id] + j;
                col_id = col_data[col_idx];

                if (col_id2idx_map.count(col_id) == 0) {
                    col_id2idx_map[col_id] = new_col_ids.size();
                    new_col_ids.push_back(col_id);
                }

                cols.push_back(col_id2idx_map[col_id]);
            }
            offset = cols.size();
            out_rowptr_data[i + 1] = offset;
        }
    } else {
        int64_t row_id, col_id, col_idx, offset = 0;
        for (int64_t i = 0; i < row_ids.size(0); i++) { //i =1024
            row_id = row_ids_data[i];
            if (rowcount_data[row_id]!=0) {
                for (int64_t j = 0; j < num_neighbors; j++) {
                    col_idx = rowptr_data[row_id] + rand() % rowcount_data[row_id];
                    col_id = col_data[col_idx];

                    if (col_id2idx_map.count(col_id) == 0) {
                        col_id2idx_map[col_id] = new_col_ids.size();
                        new_col_ids.push_back(col_id);
                    }

                    cols.push_back(col_id2idx_map[col_id]);
                }
            }
            offset = cols.size();
            out_rowptr_data[i + 1] = offset;
        }
    }

    int64_t n_len = new_col_ids.size(), e_len = cols.size();
    col = torch::from_blob(cols.data(), {e_len}, col.options()).clone();
    auto n_id = torch::from_blob(new_col_ids.data(), {n_len}, col.options()).clone();

    return std::make_tuple(out_rowptr, col, n_id);
}

void test(){
    std::cout<<"hello world";
}

PYBIND11_MODULE(hysample_cpp, m) {
    m.def("hysample_adj", &sample_adj, "hypergraph sample");
    m.def("test", &test, "test hello world");
}