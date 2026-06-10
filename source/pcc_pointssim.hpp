/* The copyright in this software is being made available under the BSD
 * License, included below. This software may be subject to other third party
 * and contributor rights, including patent rights, and no such rights are
 * granted under this license.
 */

#ifndef PCC_POINTSSIM_HPP
#define PCC_POINTSSIM_HPP

#include "pcc_processing.hpp"

namespace pcc_quality {

  class pointSSIMPar {
  public:
    pointSSIMPar();

    bool bColor;
    int neighbors;
  };

  class pointSSIMMetric {
  public:
    pointSSIMMetric();

    bool hasGeometry;
    bool hasColor;

    double geom_ba;
    double geom_ab;
    double geom_sym;

    double color_ba;
    double color_ab;
    double color_sym;
  };

  void computePointSSIM(pcc_processing::PccPointCloud& cloudA,
                        pcc_processing::PccPointCloud& cloudB,
                        const pointSSIMPar& par,
                        pointSSIMMetric& metric,
                        const bool verbose = true);

};

#endif
