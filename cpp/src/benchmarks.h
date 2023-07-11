//
// Created by Константин Носорев on 16.05.2023.
//

#ifndef DETECTOR_CPP_BENCHMARKS_H
#define DETECTOR_CPP_BENCHMARKS_H

#include "geometry/Point.h"
#include "geometry/Ellipse.h"

namespace benchmark {
    template<typename T>
    void bench_plane_algs(T functor) {
        int test_cnt = 100;
        std::vector<double> metrics_recall(test_cnt);
        std::vector<double> metrics_E(test_cnt);
        double secs = 0;
        double max_secs = 0;
        for (int cur = 0; cur < test_cnt; cur++) {
            std::stringstream filename_tmpl;
            filename_tmpl << "ellipse_samples/ellipse_" << cur << ".csv";

            auto filename = filename_tmpl.str();
            rapidcsv::Document doc(filename);
            std::vector<double> x = doc.GetColumn<double>("x");
            std::vector<double> y = doc.GetColumn<double>("y");
            std::vector<uint> signal = doc.GetColumn<uint>("signal");
            std::vector<Point> points;
            points.reserve(x.size());
            for (int i = 0; i < x.size(); i++) {
                points.emplace_back(x[i], y[i], i);
            }
            auto start = clock();
            Ellipse el;
            try {
                el = functor(points, 0.5);
            } catch (std::exception &) {
                metrics_recall[cur] = 0;
                metrics_E[cur] = std::numeric_limits<double>::infinity();
                continue;
            }
            auto cur_time = (double) (clock() - start) / CLOCKS_PER_SEC;
            secs += cur_time;
            max_secs = fmax(cur_time, max_secs);
            auto inliers = el.getInliner();
            std::vector<uint> pred_signal(x.size());
            for (const auto &i: inliers) {
                pred_signal[i.index] = 1;
            }
            metrics_recall[cur] = recall(signal, pred_signal);
            metrics_E[cur] = el.mean_distance;
            std::vector<double> metrics_col(x.size());
            std::vector<double> metrics_err(x.size());
            for (int i = 0; i < x.size(); i++) {
                metrics_col[i] = metrics_recall[cur];
                metrics_err[i] = el.mean_distance;
            }

            doc.InsertColumn(3, pred_signal, "pred");
            doc.InsertColumn(4, metrics_col, "metrics_recall");
            doc.InsertColumn(5, metrics_err, "metrics_err");
            std::stringstream out_filename_tmpl;
            out_filename_tmpl << "ellipse_samples/ellipse_pred_" << cur << ".csv";
            auto output = out_filename_tmpl.str();
            doc.Save(output);
        }
        secs /= test_cnt;
        auto recall_mean = vector_mean(metrics_recall);
        auto recall_max = vector_max(metrics_recall);
        auto recall_min = vector_min(metrics_recall);
        auto E_mean = vector_mean(metrics_E);
        auto E_max = vector_max(metrics_E);
        auto E_min = vector_min(metrics_E);
        std::cout << "Time: max " << max_secs << " mean " << secs << std::endl;
        std::cout << "Recall: max " << recall_max << " mean " << recall_mean << " min " << recall_min << std::endl;
        std::cout << "E: max " << E_max << " mean " << E_mean << " min " << E_min << std::endl;
    }
}


#endif //DETECTOR_CPP_BENCHMARKS_H
