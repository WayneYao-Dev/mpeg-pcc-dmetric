/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 *
 * Copyright (c) 2017-2025, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of the copyright holder(s) nor the names of its
 *    contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "nanoflann/nanoflann.hpp"
#include "nanoflann/KDTreeVectorOfVectorsAdaptor.h"

#include "pcc_pointssim.hpp"

using namespace std;
using namespace pcc_processing;
using namespace pcc_quality;

namespace {

  const size_t kPointSSIMExternalNeighbors = 11;
  const double kPointSSIMEpsilon = 2.2204460492503131e-16;

  typedef uint32_t index_type;
  typedef double distance_type;

  struct metric_L2_double {
    template<class T, class DataSource>
    struct traits {
      typedef nanoflann::L2_Adaptor<T, DataSource, double> distance_t;
    };
  };

  typedef KDTreeVectorOfVectorsAdaptor<
      vector<PointXYZSet::point_type>,
      PointXYZSet::point_type::value_type,
      3,
      metric_L2_double,
      index_type
      > pointSSIMKdTree;

  struct featureMaps {
    vector<double> var;
    vector<double> mean;
  };

  double sampleVariance(const vector<double>& values)
  {
    if (values.size() < 2)
      return 0.0;

    double mean = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
      mean += values[i];
    mean /= static_cast<double>(values.size());

    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
      const double difference = values[i] - mean;
      sum += difference * difference;
    }

    return sum / static_cast<double>(values.size() - 1);
  }

  double arithmeticMean(const vector<double>& values)
  {
    double sum = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
      sum += values[i];
    return sum / static_cast<double>(values.size());
  }

  double rgbToLuma(const RGBSet::value_type& rgb)
  {
    double luma = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    luma = floor(luma + 0.5);
    return max(0.0, min(255.0, luma));
  }

  bool geometryFeatureMaps(PccPointCloud& cloud, featureMaps& maps)
  {
    if (cloud.size <= static_cast<long int>(kPointSSIMExternalNeighbors))
      return false;

    const size_t neighborCount = kPointSSIMExternalNeighbors + 1;
    pointSSIMKdTree index(3, cloud.xyz.p, 10);
    maps.var.resize(static_cast<size_t>(cloud.size));
    maps.mean.resize(static_cast<size_t>(cloud.size));

#pragma omp parallel for
    for (long i = 0; i < cloud.size; ++i) {
      vector<index_type> neighborIndices(neighborCount);
      vector<distance_type> squaredDistances(neighborCount);
      index.query(&cloud.xyz.p[i][0], neighborCount,
                  &neighborIndices[0], &squaredDistances[0]);

      vector<double> distances;
      distances.reserve(kPointSSIMExternalNeighbors);
      for (size_t j = 1; j < neighborCount; ++j)
        distances.push_back(sqrt(squaredDistances[j]));

      maps.var[static_cast<size_t>(i)] = sampleVariance(distances);
      maps.mean[static_cast<size_t>(i)] = arithmeticMean(distances);
    }

    return true;
  }

  bool colorFeatureMaps(PccPointCloud& cloud, featureMaps& maps)
  {
    if (!cloud.bRgb ||
        cloud.size <= static_cast<long int>(kPointSSIMExternalNeighbors))
      return false;

    const size_t neighborCount = kPointSSIMExternalNeighbors + 1;
    pointSSIMKdTree index(3, cloud.xyz.p, 10);
    vector<double> luma(static_cast<size_t>(cloud.size));
    maps.var.resize(static_cast<size_t>(cloud.size));
    maps.mean.resize(static_cast<size_t>(cloud.size));

    for (long i = 0; i < cloud.size; ++i)
      luma[static_cast<size_t>(i)] = rgbToLuma(cloud.rgb.c[static_cast<size_t>(i)]);

#pragma omp parallel for
    for (long i = 0; i < cloud.size; ++i) {
      vector<index_type> neighborIndices(neighborCount);
      vector<distance_type> squaredDistances(neighborCount);
      index.query(&cloud.xyz.p[i][0], neighborCount,
                  &neighborIndices[0], &squaredDistances[0]);

      vector<double> values;
      values.reserve(neighborCount);
      for (size_t j = 0; j < neighborCount; ++j)
        values.push_back(luma[neighborIndices[j]]);

      maps.var[static_cast<size_t>(i)] = sampleVariance(values);
      maps.mean[static_cast<size_t>(i)] = arithmeticMean(values);
    }

    return true;
  }

  vector<size_t> nearestIndices(PccPointCloud& reference,
                                PccPointCloud& query)
  {
    pointSSIMKdTree index(3, reference.xyz.p, 10);
    vector<size_t> indices(static_cast<size_t>(query.size));

#pragma omp parallel for
    for (long i = 0; i < query.size; ++i) {
      array<index_type, 1> resultIndex;
      array<distance_type, 1> squaredDistance;
      index.query(&query.xyz.p[i][0], 1,
                  &resultIndex[0], &squaredDistance[0]);
      indices[static_cast<size_t>(i)] = resultIndex[0];
    }

    return indices;
  }

  double meanSimilarity(const vector<double>& queryFeatures,
                        const vector<double>& referenceFeatures,
                        const vector<size_t>& queryToReference)
  {
    double sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < queryFeatures.size(); ++i) {
      const double reference = referenceFeatures[queryToReference[i]];
      const double query = queryFeatures[i];
      const double error = fabs(reference - query) /
          (max(fabs(reference), fabs(query)) + kPointSSIMEpsilon);
      const double similarity = 1.0 - error;

      if (!isnan(similarity)) {
        sum += similarity;
        ++count;
      }
    }

    if (count == 0)
      return numeric_limits<double>::quiet_NaN();

    return sum / static_cast<double>(count);
  }

  void computeScores(const featureMaps& mapsA,
                     const featureMaps& mapsB,
                     const vector<size_t>& indicesBA,
                     const vector<size_t>& indicesAB,
                     pointSSIMScore& score)
  {
    score.ba.var = meanSimilarity(mapsB.var, mapsA.var, indicesBA);
    score.ab.var = meanSimilarity(mapsA.var, mapsB.var, indicesAB);
    score.sym.var = min(score.ba.var, score.ab.var);

    score.ba.mean = meanSimilarity(mapsB.mean, mapsA.mean, indicesBA);
    score.ab.mean = meanSimilarity(mapsA.mean, mapsB.mean, indicesAB);
    score.sym.mean = min(score.ba.mean, score.ab.mean);
  }

  void printScore(const char* attribute, const pointSSIMScore& score)
  {
    cout << "   " << attribute << "SSIM,VAR,BA   : " << score.ba.var << endl;
    cout << "   " << attribute << "SSIM,VAR,AB   : " << score.ab.var << endl;
    cout << "   " << attribute << "SSIM,VAR,sym  : " << score.sym.var << endl;
    cout << "   " << attribute << "SSIM,Mean,BA  : " << score.ba.mean << endl;
    cout << "   " << attribute << "SSIM,Mean,AB  : " << score.ab.mean << endl;
    cout << "   " << attribute << "SSIM,Mean,sym : " << score.sym.mean << endl;
  }

}

pointSSIMEstimatorScore::pointSSIMEstimatorScore()
  : var(0.0), mean(0.0)
{}

pointSSIMMetric::pointSSIMMetric()
  : hasGeometry(false), hasColor(false)
{}

void pcc_quality::computePointSSIM(PccPointCloud& cloudA,
                                   PccPointCloud& cloudB,
                                   bool computeGeometry,
                                   bool computeColor,
                                   pointSSIMMetric& metric,
                                   const bool verbose)
{
  if (cloudA.size == 0 || cloudB.size == 0)
    return;

  if (cloudA.size <= static_cast<long int>(kPointSSIMExternalNeighbors) ||
      cloudB.size <= static_cast<long int>(kPointSSIMExternalNeighbors)) {
    if (verbose) {
      cout << "WARNING: PointSSIM requires each point cloud to contain more "
              "than 11 points.\n";
    }
    return;
  }

  if (computeColor && (!cloudA.bRgb || !cloudB.bRgb)) {
    if (verbose) {
      cout << "WARNING: no color properties in input files, disabling "
              "PointSSIM color metrics.\n";
    }
    computeColor = false;
  }

  if (!computeGeometry && !computeColor)
    return;

  const vector<size_t> indicesBA = nearestIndices(cloudA, cloudB);
  const vector<size_t> indicesAB = nearestIndices(cloudB, cloudA);

  if (computeGeometry) {
    featureMaps mapsA;
    featureMaps mapsB;
    if (geometryFeatureMaps(cloudA, mapsA) &&
        geometryFeatureMaps(cloudB, mapsB)) {
      computeScores(mapsA, mapsB, indicesBA, indicesAB, metric.geometry);
      metric.hasGeometry = true;
    }
  }

  if (computeColor) {
    featureMaps mapsA;
    featureMaps mapsB;
    if (colorFeatureMaps(cloudA, mapsA) &&
        colorFeatureMaps(cloudB, mapsB)) {
      computeScores(mapsA, mapsB, indicesBA, indicesAB, metric.color);
      metric.hasColor = true;
    }
  }

  if (verbose && (metric.hasGeometry || metric.hasColor)) {
    cout << "4. PointSSIM.\n";
    if (metric.hasGeometry)
      printScore("geo", metric.geometry);
    if (metric.hasColor)
      printScore("color", metric.color);
  }
}
