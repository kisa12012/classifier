#include <confidence_weighted/cw.h>

#include <cmath>
#include <cfloat>
#include <algorithm>

namespace classifier {
namespace cw {
CW::CW(double phi) : mode_(0), phi_(phi), C_(1.0) {
  weight_matrix().swap(weight_);
  covariance_matrix().swap(cov_);
}

void CW::Train(const datum& datum) {
  if (mode_ != 0 && mode_ != 1 && mode_ != 2) {
    std::cout << "mode should be set {0,1,2}." << std::endl;
    std::cout << " mode 0 : Confidence-Weighted" << std::endl;
    std::cout << " mode 1 : Soft Confidence-Weighted1" << std::endl;
    std::cout << " mode 2 : Soft Confidence-Weighted2" << std::endl;
    return;
  }
  score2class scores(0);
  CalcScores(datum.fv, &scores);
  Update(datum, scores);
}

void CW::Train(const std::vector<datum>& data,
               const size_t iteration) {
  for (size_t iter = 0; iter < iteration; ++iter) {
    for (std::vector<datum>::const_iterator it = data.begin();
         it != data.end();
         ++it) {
      Train(*it);
    }
  }
}

void CW::Test(const feature_vector& fv,
              std::string* predict) const {
  score2class scores(0);
  CalcScores(fv, &scores);
  *predict = scores[0].second;
}

void CW::CalcScores(const feature_vector& fv,
                    score2class* scores) const {
  scores->push_back(make_pair(non_class_score, non_class));

  for (weight_matrix::const_iterator it = weight_.begin();
       it != weight_.end();
       ++it) {
    double score = InnerProduct(fv, it->second);
    scores->push_back(make_pair(score, it->first));
  }

  sort(scores->begin(), scores->end(),
       std::greater<std::pair<double, std::string> >());
}

double CW::CalcV(const datum& datum,
                 const std::string& non_correct_predict) {
  double v = 0.0;
  covariance_vector &correct_cov = cov_[datum.category];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (correct_cov.size() <= it->first)
      correct_cov.resize(it->first + 1, 1.0);
    v += correct_cov[it->first] * it->second * it->second;
  }

  if (non_correct_predict == non_class)
    return v;

  covariance_vector &wrong_cov = cov_[non_correct_predict];
  for (feature_vector::const_iterator it = datum.fv.begin();
       it != datum.fv.end();
       ++it) {
    if (wrong_cov.size() <= it->first)
      wrong_cov.resize(it->first + 1, 1.0);
    v += wrong_cov[it->first] * it->second * it->second;
  }

  return v;
}

double CW::CalcAlpha(double m, double v) const {
  if (mode_ == 0) {
    return CalcAlpha0(m, v);
  } else if (mode_ == 1) {
    return CalcAlpha1(m, v);
  } else if (mode_ == 2) {
    return CalcAlpha2(m, v);
  }
  return 0.0;
}

double CW::CalcAlpha0(double m, double v) const {
  double psi = 1.0 + phi_ * phi_ / 2.0;
  double zeta = 1.0 + phi_ * phi_;

  double alpha =
    (-m * psi + std::sqrt(m*m*phi_*phi_/4.0 + v*phi_*phi_*zeta)) / (v * zeta);
  return std::max(0.0, alpha);
}

double CW::CalcAlpha1(double m, double v) const {
  return std::min(C_, CalcAlpha0(m, v));
}

double CW::CalcAlpha2(double m, double v) const {
  double n = v + 1.0 / (2.0 * C_);
  double gamma = phi_ * std::sqrt(phi_*phi_*m*m*v*v + 4*n*v*(n+v*phi_*phi_));
  double alpha = - (2.0 * m * n + phi_ * phi_ * m * v) + gamma;
  if (alpha <= 0.0) return 0.0;
  alpha /= 2.0 * (n*n + n*v*phi_*phi_);
  return alpha;
}

double CW::CalcBeta(double v, double alpha) const {
  double u = (-alpha * v * phi_
              + std::sqrt(alpha * alpha * v * v * phi_ * phi_ + 4.0 * v));
  u = u * u / 4.0;

  double beta = alpha * phi_ / (std::sqrt(u) + v * alpha * phi_);
  return beta;
}

void CW::Update(const datum& datum,
                const score2class& scores) {
  std::string non_correct_predict;
  double m = - CalcLossScore(scores, datum.category, &non_correct_predict);
  double v = CalcV(datum, non_correct_predict);
  double alpha = CalcAlpha(m, v);
  double beta = CalcBeta(v, alpha);

  if (alpha > 0.0) {
    weight_vector &correct_weight = weight_[datum.category];
    covariance_vector &correct_cov = cov_[datum.category];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (correct_weight.size() <= it->first)
        correct_weight.resize(it->first + 1, 0.0);
      correct_weight[it->first] += alpha * correct_cov[it->first] * it->second;

      correct_cov[it->first] -=
        beta * it->second * it->second * correct_cov[it->first] * correct_cov[it->first];
    }

    if (non_correct_predict == non_class)
      return;

    weight_vector &wrong_weight = weight_[non_correct_predict];
    covariance_vector &wrong_cov = cov_[non_correct_predict];
    for (feature_vector::const_iterator it = datum.fv.begin();
         it != datum.fv.end();
         ++it) {
      if (wrong_weight.size() <= it->first)
        wrong_weight.resize(it->first + 1, 0.0);
      wrong_weight[it->first] -= alpha * wrong_cov[it->first] * it->second;

      wrong_cov[it->first] +=
        beta * it->second * it->second * correct_cov[it->first] * correct_cov[it->first];
    }
  }
}

void CW::GetFeatureWeight(size_t feature_id,
                          std::vector<std::pair<std::string, double> >* results) const {
  ReturnFeatureWeight(feature_id, weight_, results);
}

} //namespace
} //namespace
