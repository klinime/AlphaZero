#ifndef OTHELLO_HPP
#define OTHELLO_HPP

#include "mcts.hpp"

namespace othello {
    class Game : public base::Game {
        public:
        Game();
        Game(uint8_t);
        uint8_t get_turn();
        int get_board_dim();
        int get_board_size();
        int get_ac_dim();
        int get_state_depth();
        std::vector<uint8_t> get_state();
        std::vector<uint16_t> get_action();
        std::unique_ptr<base::Game> take_action(uint16_t);
        int terminal();
        int evaluate();
        std::string state_str();
        std::vector<std::string> action_str();
        
        protected:
        uint64_t m_b_board;
        uint64_t m_w_board;
        uint8_t  m_pass_count;
        std::vector<uint8_t> m_state;
        std::vector<uint16_t> m_actions;
        uint8_t m_state_init;
    };
}

#endif
