//
// Created by 朱世杨 on 2020/8/20.
//

#ifndef EXAMPLE_APP_HYSAMPLE_H
#define EXAMPLE_APP_HYSAMPLE_H

#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
sample_adj(torch::Tensor rowptr, torch::Tensor col, torch::Tensor rowcount,
           torch::Tensor row_ids, torch::Tensor col_ids, int64_t num_neighbors);

#endif //EXAMPLE_APP_HYSAMPLE_H
