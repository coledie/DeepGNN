// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_CLIENT_H
#define SNARK_CLIENT_H

#include <atomic>
#include <filesystem>
#include <functional>
#include <mutex>
#include <span>
#include <string>
#include <thread>

#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/completion_queue.h>

#include "src/cc/lib/distributed/service.grpc.pb.h"
#include "src/cc/lib/graph/graph.h"

namespace snark
{

class GRPCClient final
{
  public:
    GRPCClient(std::vector<std::shared_ptr<grpc::Channel>> channels, uint32_t num_threads, uint32_t num_threads_per_cq);
    void GetNodeType(std::span<const NodeId> node_ids, std::span<Type> output, Type default_type,
                     std::string graph_id = "0");
    void GetNodeFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                        std::span<FeatureMeta> features, std::span<uint8_t> output, std::string graph_id = "0");

    void GetEdgeFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                        std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                        std::span<FeatureMeta> features, std::span<uint8_t> output, std::string graph_id = "0");

    void GetNodeSparseFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                              std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                              std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values, std::string graph_id = "0");

    void GetEdgeSparseFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                              std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                              std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                              std::vector<std::vector<int64_t>> &out_indices,
                              std::vector<std::vector<uint8_t>> &out_values, std::string graph_id = "0");

    void GetNodeStringFeature(std::span<const NodeId> node_ids, std::span<const snark::Timestamp> timestamps,
                              std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                              std::vector<uint8_t> &out_values, std::string graph_id = "0");

    void GetEdgeStringFeature(std::span<const NodeId> edge_src_ids, std::span<const NodeId> edge_dst_ids,
                              std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                              std::span<const FeatureId> features, std::span<int64_t> out_dimensions,
                              std::vector<uint8_t> &out_values, std::string graph_id = "0");

    void NeighborCount(std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                       std::span<const snark::Timestamp> timestamps, std::span<uint64_t> output_neighbor_counts,
                       std::string graph_id = "0");

    void FullNeighbor(bool return_edge_created_ts, std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                      std::span<const snark::Timestamp> timestamps, std::vector<NodeId> &output_nodes,
                      std::vector<Type> &output_types, std::vector<float> &output_weights,
                      std::vector<Timestamp> &out_edge_created_ts, std::span<uint64_t> output_neighbor_counts,
                      std::string graph_id = "0");

    void WeightedSampleNeighbor(bool return_edge_created_ts, int64_t seed, std::span<const NodeId> node_ids,
                                std::span<const Type> edge_types, std::span<const snark::Timestamp> timestamps,
                                size_t count, std::span<NodeId> output_nodes, std::span<Type> output_types,
                                std::span<float> output_weights, std::span<Timestamp> output_edge_created_ts,
                                NodeId default_node_id, float default_weight, Type default_edge_type,
                                std::string graph_id = "0");

    void UniformSampleNeighbor(bool without_replacement, bool return_edge_created_ts, int64_t seed,
                               std::span<const NodeId> node_ids, std::span<const Type> edge_types,
                               std::span<const snark::Timestamp> timestamps, size_t count,
                               std::span<NodeId> output_nodes, std::span<Type> output_types,
                               std::span<Timestamp> output_edge_created_ts, NodeId default_node_id, Type default_type,
                               std::string graph_id = "0");

    void LastNCreated(bool return_edge_created_ts, std::span<const NodeId> input_node_ids,
                      std::span<Type> input_edge_types, std::span<const Timestamp> input_timestamps, size_t count,
                      std::span<NodeId> output_neighbor_ids, std::span<Type> output_neighbor_types,
                      std::span<float> neighbors_weights, std::span<Timestamp> output_timestamps,
                      NodeId default_node_id, float default_weight, Type default_edge_type, Timestamp default_timestamp,
                      std::string graph_id = "0");

    uint64_t CreateSampler(bool is_edge, CreateSamplerRequest_Category category, std::span<Type> types,
                           std::string graph_id = "0");

    void SampleNodes(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_node_ids, std::span<Type> output_types,
                     std::string graph_id = "0");

    void SampleEdges(int64_t seed, uint64_t sampler_id, std::span<NodeId> out_src_node_ids,
                     std::span<Type> output_types, std::span<NodeId> out_dst_node_ids, std::string graph_id = "0");
    void WriteMetadata(std::filesystem::path path, std::string graph_id = "0");

    ~GRPCClient();

  private:
    std::mutex m_sampler_mutex;
    std::vector<std::vector<uint64_t>> m_sampler_ids;
    std::vector<std::vector<float>> m_sampler_weights;

    std::function<void()> AsyncCompleteRpc(size_t i);
    grpc::CompletionQueue *NextCompletionQueue();

    std::vector<std::unique_ptr<GraphEngine::Stub>> m_engine_stubs;
    std::vector<std::unique_ptr<GraphSampler::Stub>> m_sampler_stubs;
    std::vector<grpc::CompletionQueue> m_completion_queue;
    std::vector<std::thread> m_reply_threads;
    std::atomic<size_t> m_counter;
};

} // namespace snark
#endif // SNARK_CLIENT_H
