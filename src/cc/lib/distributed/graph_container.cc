// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "graph_container.h"

#include <cstdio>
#include <limits>

#include <glog/logging.h>
#include <glog/raw_logging.h>

namespace snark
{

GraphContainer::GraphContainer(std::unordered_map<std::string, std::shared_ptr<snark::GraphEngine::Service>> container)
    : m_container(std::move(container))
{
}

GraphContainer::~GraphContainer()
{
    return;
}

grpc::Status GraphContainer::GetNodeTypes(::grpc::ServerContext *context, const snark::NodeTypesRequest *request,
                                          snark::NodeTypesReply *response)
{
    return m_container.find("0")->second->GetNodeTypes(context, request, response);
}

grpc::Status GraphContainer::GetNodeFeatures(::grpc::ServerContext *context, const snark::NodeFeaturesRequest *request,
                                             snark::NodeFeaturesReply *response)
{
    return m_container.find("0")->second->GetNodeFeatures(context, request, response);
}

grpc::Status GraphContainer::GetEdgeFeatures(::grpc::ServerContext *context, const snark::EdgeFeaturesRequest *request,
                                             snark::EdgeFeaturesReply *response)
{
    return m_container.find("0")->second->GetEdgeFeatures(context, request, response);
}

grpc::Status GraphContainer::GetNodeSparseFeatures(::grpc::ServerContext *context,
                                                   const snark::NodeSparseFeaturesRequest *request,
                                                   snark::SparseFeaturesReply *response)
{
    return m_container.find("0")->second->GetNodeSparseFeatures(context, request, response);
}

grpc::Status GraphContainer::GetEdgeSparseFeatures(::grpc::ServerContext *context,
                                                   const snark::EdgeSparseFeaturesRequest *request,
                                                   snark::SparseFeaturesReply *response)
{
    return m_container.find("0")->second->GetEdgeSparseFeatures(context, request, response);
}

grpc::Status GraphContainer::GetNodeStringFeatures(::grpc::ServerContext *context,
                                                   const snark::NodeSparseFeaturesRequest *request,
                                                   snark::StringFeaturesReply *response)
{
    return m_container.find("0")->second->GetNodeStringFeatures(context, request, response);
}

grpc::Status GraphContainer::GetEdgeStringFeatures(::grpc::ServerContext *context,
                                                   const snark::EdgeSparseFeaturesRequest *request,
                                                   snark::StringFeaturesReply *response)
{
    return m_container.find("0")->second->GetEdgeStringFeatures(context, request, response);
}

grpc::Status GraphContainer::GetNeighborCounts(::grpc::ServerContext *context,
                                               const snark::GetNeighborsRequest *request,
                                               snark::GetNeighborCountsReply *response)
{
    return m_container.find("0")->second->GetNeighborCounts(context, request, response);
}

grpc::Status GraphContainer::GetNeighbors(::grpc::ServerContext *context, const snark::GetNeighborsRequest *request,
                                          snark::GetNeighborsReply *response)
{
    return m_container.find("0")->second->GetNeighbors(context, request, response);
}

grpc::Status GraphContainer::GetLastNCreatedNeighbors(::grpc::ServerContext *context,
                                                      const snark::GetLastNCreatedNeighborsRequest *request,
                                                      snark::GetNeighborsReply *response)
{
    return m_container.find("0")->second->GetLastNCreatedNeighbors(context, request, response);
}

grpc::Status GraphContainer::WeightedSampleNeighbors(::grpc::ServerContext *context,
                                                     const snark::WeightedSampleNeighborsRequest *request,
                                                     snark::WeightedSampleNeighborsReply *response)
{
    return m_container.find("0")->second->WeightedSampleNeighbors(context, request, response);
}

grpc::Status GraphContainer::UniformSampleNeighbors(::grpc::ServerContext *context,
                                                    const snark::UniformSampleNeighborsRequest *request,
                                                    snark::UniformSampleNeighborsReply *response)
{
    return m_container.find("0")->second->UniformSampleNeighbors(context, request, response);
}

grpc::Status GraphContainer::GetMetadata(::grpc::ServerContext *context, const snark::EmptyMessage *request,
                                         snark::MetadataReply *response)
{
    return m_container.find("0")->second->GetMetadata(context, request, response);
}

} // namespace snark
