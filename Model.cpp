#include "Model.h"

#include <stdexcept>
#include <iostream>
#include <unordered_map>

#include "TOLInclude.h"
#include "Utils.h"

Model::Model(const char *fileName) : vertices(), indices() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, fileName)) {
        throw std::runtime_error("Error on loading model: " + warn + err);
    }
    if (!warn.empty()) {
        std::cerr << print_time << "Warns on loading \"" << fileName << "\": " << warn << std::endl;
    }
    if (!err.empty()) {
        std::cerr << print_time << "Errors on loading \"" << fileName << "\": " << err << std::endl;
    }

    size_t totalVerticesCount = 0;
    for (const auto &shape : shapes) {
        totalVerticesCount += shape.mesh.indices.size();
    }

    vertices.reserve(totalVerticesCount);
    indices.reserve(totalVerticesCount);

    std::unordered_map<so::Vertex, uint32_t> uniqueVertices{};
    uniqueVertices.reserve(totalVerticesCount);

    for (const auto &shape : shapes) {
        for (const auto &index : shape.mesh.indices) {
            so::Vertex vertex {
                    {
                            attrib.vertices[3 * index.vertex_index + 0],
                            attrib.vertices[3 * index.vertex_index + 1],
                            attrib.vertices[3 * index.vertex_index + 2]
                    },
                    {1.0f, 1.0f, 1.0f},
                    {
                            attrib.texcoords[2 * index.texcoord_index + 0],
                            1.0f - attrib.texcoords[2 * index.texcoord_index + 1]
                    }
            };

            const auto it = uniqueVertices.find(vertex);
            if (it == uniqueVertices.cend()) {
                const auto currentVerticesCount = vertices.size();
                indices.push_back(currentVerticesCount);
                uniqueVertices[vertex] = currentVerticesCount;
                vertices.emplace_back(vertex);
            } else {
                indices.push_back(it->second);
            }
        }
    }

    vertices.shrink_to_fit();
}
