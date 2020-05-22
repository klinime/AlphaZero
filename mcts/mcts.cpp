#include <algorithm>
#include <random>
#include <cmath>
#include "mcts.hpp"

namespace util {
    std::default_random_engine gen(0);

    template <typename T>
    void softmax(T beg, T end) {
        using VType = typename std::iterator_traits<T>::value_type;
        VType max_e = *std::max_element(beg, end);
        std::transform(beg, end, beg, [&](VType x) { return std::exp(x - max_e); });
        VType sum_e = std::accumulate(beg, end, 0.0);
        std::transform(beg, end, beg, [&](VType val) { return val / sum_e; });
    }

    template <typename T>
    void add_dirichlet_noise(T beg, T end, float alpha, float epsilon) {
        using VType = typename std::iterator_traits<T>::value_type;
        std::gamma_distribution<VType> gamma(alpha, 1);
        std::vector<VType> noise(end - beg);
        std::generate(noise.begin(), noise.end(), [&]() { return gamma(gen); });
        VType sum_n = std::accumulate(noise.begin(), noise.end(), 0.0);
        std::transform(noise.begin(), noise.end(), noise.begin(),
            [&](VType val) { return val / sum_n; });
        std::transform(beg, end, noise.begin(), beg,
            [&](VType x, VType y) { return (1 - epsilon) * x + epsilon * y; });
        VType sum_v = std::accumulate(beg, end, 0.0);
        std::transform(beg, end, beg, [&](VType val) { return val / sum_v; });
    }
}

namespace mcts {
    
    std::vector<std::vector<uint8_t>> get_eval_states(
            std::vector<std::shared_ptr<Node>> &chosen, int history) {
        // stack history amount of states
        std::vector<std::vector<uint8_t>> states(chosen.size() * history);
        for (size_t i = 0; i < chosen.size(); ++i) {
            auto t = chosen[i];
            int offset = history - 1;
            while (offset >= 0 && t) {
                states[i * history + offset] = t->m_game->get_state();
                t = t->m_parent.lock();
                --offset;
            }
            if (offset >= 0) {
                int size = chosen[0]->m_game->board_height() * \
                    chosen[0]->m_game->board_width() * \
                    chosen[0]->m_game->state_depth();
                std::vector<uint8_t> state(size);
                while (offset >= 0) {
                    states[i * history + offset] = state;
                    --offset;
                }
            }
        }
        return states;
    }

    void select(std::vector<std::shared_ptr<Node>> &trees,
                std::vector<std::shared_ptr<Node>> &chosen,
                int tau, float c_puct, float alpha, float epsilon, int cutoff) {
        // select nodes from trees and put into chosen
        for (int i = trees.size() - 1; i >= 0; --i) {
            auto t = trees[i];
            while (t->m_game->terminal() & (t->m_children.size() > 0)) {
                // add dirichlet noise for exploration
                std::vector<float> prior = t->m_prior;
                if ((t == trees[i]) & (t->m_game->get_turn() <= cutoff) & (tau > 0)) {
                    util::add_dirichlet_noise(
                        prior.begin(), prior.end(), alpha, epsilon);
                }

                // choose the argmax of children's scores
                std::vector<float> scores(t->m_children.size());
                for (int j = t->m_children.size() - 1; j >= 0; --j) {
                    auto child = t->m_children[j];
                    scores[j] = (1 - child->m_value) + c_puct * prior[j] * 
                        sqrt(t->m_visit - 1) / (1 + child->m_visit);
                }
                int max_idx = static_cast<int>(std::distance(scores.begin(), 
                    std::max_element(scores.begin(), scores.end())));
                t = t->m_children[max_idx];
            }
            chosen[i] = t;

            // inintialize children of chosen node
            if (t->m_game->terminal()) {
                auto actions = t->m_game->get_action();
                t->m_children.reserve(actions.size());
                for (size_t j = 0; j < actions.size(); ++j) {
                    t->m_children.push_back(std::make_shared<Node>(
                        t, std::move(t->m_game->take_action(actions[j]))));
                }
            }
        }
    }

    std::vector<float> expand(std::vector<std::shared_ptr<Node>> &chosen) {
        // random simulation - default mcts evaluation
        std::vector<float> values(chosen.size());
        for (int i = chosen.size() - 1; i >= 0; --i) {
            auto actions = chosen[i]->m_game->get_action();
            auto game = chosen[i]->m_game->take_action(
                actions[rand() % actions.size()]);
            while (game->terminal()) {
                actions = game->get_action();
                game = game->take_action(actions[rand() % actions.size()]);
            }
            values[i] = !((game->get_turn() & 1) ^ ((game->evaluate() + 1) >> 1));
        }
        return values;
    }

    void backprop(std::vector<std::shared_ptr<Node>> &trees,
                  std::vector<std::shared_ptr<Node>> &chosen,
                  std::vector<float> &priors, std::vector<float> &values) {
        // assign priors to chosen nodes and backprop values until trees (root)
        int ac_dim = trees[0]->m_game->ac_dim();
        for (int i = chosen.size() - 1; i >= 0; --i) {
            auto child = chosen[i];
            auto actions = child->m_game->get_action();

            // filter prior with legal actions and softmax
            child->m_prior.reserve(actions.size());
            for (size_t j = 0; j < actions.size(); ++j) {
                child->m_prior.push_back(priors[i * ac_dim + actions[j]]);
            }
            util::softmax(child->m_prior.begin(), child->m_prior.end());

            // value conevert to [0, 1]
            float value = child->m_game->terminal() ? (values[i] + 1) / 2 :
                !((child->m_game->get_turn() & 1) ^
                  ((child->m_game->evaluate() + 1) >> 1));
            child->m_value = value;
            value = 1 - value;

            // propagate value until hitting trees (root)
            while (child != trees[i]) {
                auto parent = child->m_parent.lock();
                parent->m_value = (parent->m_value * parent->m_visit + value) /
                    (parent->m_visit + 1);
                ++parent->m_visit;
                child = parent;
                value = 1 - value;
            }
        }
    }
}
