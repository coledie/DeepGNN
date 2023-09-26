// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#ifndef SNARK_CONTAINER_H
#define SNARK_CONTAINER_H

#include <memory>
#include <thread>

#include "src/cc/lib/distributed/graph_engine.h"

namespace snark
{
class GraphContainer final
{
  public:
    GraphContainer(std::unordered_map<std::string, std::shared_ptr<snark::GraphEngine::Service>> container);

    ~GraphContainer();

    grpc::Status GetNodeTypes(::grpc::ServerContext *context, const snark::NodeTypesRequest *request,
                              snark::NodeTypesReply *response);
    grpc::Status GetNodeFeatures(::grpc::ServerContext *context, const snark::NodeFeaturesRequest *request,
                                 snark::NodeFeaturesReply *response);
    grpc::Status GetEdgeFeatures(::grpc::ServerContext *context, const snark::EdgeFeaturesRequest *request,
                                 snark::EdgeFeaturesReply *response);
    grpc::Status GetNodeSparseFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response);
    grpc::Status GetEdgeSparseFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::SparseFeaturesReply *response);
    grpc::Status GetNodeStringFeatures(::grpc::ServerContext *context, const snark::NodeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response);
    grpc::Status GetEdgeStringFeatures(::grpc::ServerContext *context, const snark::EdgeSparseFeaturesRequest *request,
                                       snark::StringFeaturesReply *response);
    grpc::Status GetNeighborCounts(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                                   snark::GetNeighborCountsReply *response);
    grpc::Status GetNeighbors(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                              snark::GetNeighborsReply *response);
    grpc::Status GetLastNCreatedNeighbors(::grpc::ServerContext *context,
                                          const snark::GetLastNCreatedNeighborsRequest *request,
                                          snark::GetNeighborsReply *response);
    grpc::Status WeightedSampleNeighbors(::grpc::ServerContext *context,
                                         const snark::WeightedSampleNeighborsRequest *request,
                                         snark::WeightedSampleNeighborsReply *response);
    grpc::Status UniformSampleNeighbors(::grpc::ServerContext *context,
                                        const snark::UniformSampleNeighborsRequest *request,
                                        snark::UniformSampleNeighborsReply *response);
    grpc::Status GetMetadata(::grpc::ServerContext *context, const snark::EmptyMessage *request,
                             snark::MetadataReply *response);

    std::unordered_map<std::string, std::shared_ptr<snark::GraphEngine::Service>> m_container;

  private:
};
} // namespace snark
#endif // SNARK_SERVER_H
