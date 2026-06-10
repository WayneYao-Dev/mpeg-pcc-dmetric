/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
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

  const double kPointSSIMEps = 2.2204460492503131e-16;

  typedef uint32_t index_type;
  typedef double distance_type;

  struct metric_L2_double {
    template<class T, class DataSource>
    struct traits {
      typedef nanoflann::L2_Adaptor<T,DataSource,double> distance_t;
    };
  };

  typedef KDTreeVectorOfVectorsAdaptor<
      vector<PointXYZSet::point_type>,
      PointXYZSet::point_type::value_type,
      3,
      metric_L2_double,
      index_type
      > my_kd_tree_t;

  double sampleVariance(const vector<double>& values)
  {
    if (values.size() < 2)
      return 0.0;

    double mean = 0.0;
    for (size_t i = 0; i < values.size(); ++i)
      mean += values[i];
    mean /= static_cast<double>(values.size());

    double accum = 0.0;
    for (size_t i = 0; i < values.size(); ++i) {
      const double diff = values[i] - mean;
      accum += diff * diff;
    }

    return accum / static_cast<double>(values.size() - 1);
  }

  double rgbToLuma(const RGBSet::value_type& rgb)
  {
    double y = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    y = floor(y + 0.5);
    if (y < 0.0)
      return 0.0;
    if (y > 255.0)
      return 255.0;
    return y;
  }

  bool geometryFeatureMapVAR(PccPointCloud& cloud,
                             const size_t externalNeighbors,
                             vector<double>& featureMap)
  {
    if (cloud.size <= static_cast<long int>(externalNeighbors))
      return false;

    const size_t knnK = externalNeighbors + 1;
    my_kd_tree_t index(3, cloud.xyz.p, 10);
    featureMap.resize(static_cast<size_t>(cloud.size));

#pragma omp parallel for
    for (long i = 0; i < cloud.size; ++i) {
      vector<index_type> indices(knnK);
      vector<distance_type> sqrDist(knnK);
      index.query(&cloud.xyz.p[i][0], knnK, &indices[0], &sqrDist[0]);

      vector<double> distances;
      distances.reserve(externalNeighbors);
      for (size_t j = 1; j < knnK; ++j)
        distances.push_back(sqrt(sqrDist[j]));

      featureMap[static_cast<size_t>(i)] = sampleVariance(distances);
    }

    return true;
  }

  bool colorFeatureMapVAR(PccPointCloud& cloud,
                          const size_t externalNeighbors,
                          vector<double>& featureMap)
  {
    if (!cloud.bRgb || cloud.size <= static_cast<long int>(externalNeighbors))
      return false;

    const size_t knnK = externalNeighbors + 1;
    my_kd_tree_t index(3, cloud.xyz.p, 10);
    vector<double> luma(static_cast<size_t>(cloud.size));
    featureMap.resize(static_cast<size_t>(cloud.size));

    for (long i = 0; i < cloud.size; ++i)
      luma[static_cast<size_t>(i)] = rgbToLuma(cloud.rgb.c[static_cast<size_t>(i)]);

#pragma omp parallel for
    for (long i = 0; i < cloud.size; ++i) {
      vector<index_type> indices(knnK);
      vector<distance_type> sqrDist(knnK);
      index.query(&cloud.xyz.p[i][0], knnK, &indices[0], &sqrDist[0]);

      vector<double> values;
      values.reserve(knnK);
      for (size_t j = 0; j < knnK; ++j)
        values.push_back(luma[indices[j]]);

      featureMap[static_cast<size_t>(i)] = sampleVariance(values);
    }

    return true;
  }

  vector<size_t> nearestIndices(PccPointCloud& reference, PccPointCloud& query)
  {
    my_kd_tree_t index(3, reference.xyz.p, 10);
    vector<size_t> indices(static_cast<size_t>(query.size));

#pragma omp parallel for
    for (long i = 0; i < query.size; ++i) {
      const size_t numResults = 1;
      array<index_type,numResults> resultIndex;
      array<distance_type,numResults> resultDist;
      index.query(&query.xyz.p[i][0], numResults, &resultIndex[0], &resultDist[0]);
      indices[static_cast<size_t>(i)] = resultIndex[0];
    }

    return indices;
  }

  double meanSimilarity(const vector<double>& yFeatureMap,
                        const vector<double>& xFeatureMap,
                        const vector<size_t>& idYX)
  {
    double sum = 0.0;
    size_t count = 0;

    for (size_t i = 0; i < yFeatureMap.size(); ++i) {
      const double x = xFeatureMap[idYX[i]];
      const double y = yFeatureMap[i];
      const double error = fabs(x - y) / (std::max(fabs(x), fabs(y)) + kPointSSIMEps);
      const double similarity = 1.0 - error;

      if (!isnan(similarity)) {
        sum += similarity;
        count++;
      }
    }

    if (count == 0)
      return numeric_limits<double>::quiet_NaN();

    return sum / static_cast<double>(count);
  }

  void computeDirectionalScores(const vector<double>& featureA,
                                const vector<double>& featureB,
                                PccPointCloud& cloudA,
                                PccPointCloud& cloudB,
                                double& scoreBA,
                                double& scoreAB,
                                double& scoreSym)
  {
    const vector<size_t> idBA = nearestIndices(cloudA, cloudB);
    const vector<size_t> idAB = nearestIndices(cloudB, cloudA);

    scoreBA = meanSimilarity(featureB, featureA, idBA);
    scoreAB = meanSimilarity(featureA, featureB, idAB);
    scoreSym = std::min(scoreBA, scoreAB);
  }

  void printPointSSIM(const pointSSIMMetric& metric)
  {
    if (metric.hasGeometry) {
      cout << "   geoSSIM,BA        : " << metric.geom_ba << endl;
      cout << "   geoSSIM,AB        : " << metric.geom_ab << endl;
      cout << "   geoSSIM,sym       : " << metric.geom_sym << endl;
    }

    if (metric.hasColor) {
      cout << "   colorSSIM,BA      : " << metric.color_ba << endl;
      cout << "   colorSSIM,AB      : " << metric.color_ab << endl;
      cout << "   colorSSIM,sym     : " << metric.color_sym << endl;
    }
  }

}

pointSSIMPar::pointSSIMPar()
{
  bColor = false;
  neighbors = 11;
}

pointSSIMMetric::pointSSIMMetric()
{
  hasGeometry = false;
  hasColor = false;

  geom_ba = 0.0;
  geom_ab = 0.0;
  geom_sym = 0.0;

  color_ba = 0.0;
  color_ab = 0.0;
  color_sym = 0.0;
}

void pcc_quality::computePointSSIM(PccPointCloud& cloudA,
                                   PccPointCloud& cloudB,
                                   const pointSSIMPar& par,
                                   pointSSIMMetric& metric,
                                   const bool verbose)
{
  if (cloudA.size == 0 || cloudB.size == 0)
    return;

  const size_t externalNeighbors = par.neighbors > 0 ? static_cast<size_t>(par.neighbors) : 11;

  vector<double> geomA;
  vector<double> geomB;
  if (geometryFeatureMapVAR(cloudA, externalNeighbors, geomA) &&
      geometryFeatureMapVAR(cloudB, externalNeighbors, geomB)) {
    computeDirectionalScores(geomA, geomB, cloudA, cloudB,
                             metric.geom_ba, metric.geom_ab, metric.geom_sym);
    metric.hasGeometry = true;
  } else if (verbose) {
    cout << "WARNING: not enough points for PointSSIM geometry metrics.\n";
  }

  if (par.bColor) {
    if (!cloudA.bRgb || !cloudB.bRgb) {
      if (verbose)
        cout << "WARNING: no color properties in input files, disabling PointSSIM color metrics.\n";
    } else {
      vector<double> colorA;
      vector<double> colorB;
      if (colorFeatureMapVAR(cloudA, externalNeighbors, colorA) &&
          colorFeatureMapVAR(cloudB, externalNeighbors, colorB)) {
        computeDirectionalScores(colorA, colorB, cloudA, cloudB,
                                 metric.color_ba, metric.color_ab, metric.color_sym);
        metric.hasColor = true;
      } else if (verbose) {
        cout << "WARNING: not enough points for PointSSIM color metrics.\n";
      }
    }
  }

  if (verbose && (metric.hasGeometry || metric.hasColor)) {
    cout << "4. PointSSIM.\n";
    printPointSSIM(metric);
  }
}
