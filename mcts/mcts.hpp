#ifndef MCTS_HPP
#define MCTS_HPP

#include <vector>
#include <string>
#include <memory>

namespace base {
    class Game {
        public:
        Game() {};
        virtual ~Game() {};
        uint8_t get_turn() { return this->m_turn; }
        virtual int board_height() { return 0; }
        virtual int board_width()  { return 0; }
        virtual int ac_dim()       { return 0; }
        virtual int state_depth()  { return 0; }
        virtual int const_depth()  { return 0; }
        virtual std::vector<uint8_t> get_state() { return std::vector<uint8_t>(); }
        virtual std::vector<uint8_t> get_const() { return std::vector<uint8_t>(); }
        virtual std::vector<uint16_t> get_action()
        { return std::vector<uint16_t>(); }
        virtual std::unique_ptr<Game> take_action(uint16_t idx) { return nullptr; }
        virtual int terminal() { return 0; }
        virtual int evaluate() { return 0; }
        virtual std::string state_str() { return std::string(); }
        virtual std::vector<std::string> action_str()
        { return std::vector<std::string>(); }
        
        protected:
        uint8_t m_turn;
    };
}

namespace mcts {
    class Node {
        public:
        std::weak_ptr<Node> m_parent;
        std::vector<std::shared_ptr<Node>> m_children;
        std::shared_ptr<base::Game> m_game;
        int   m_visit;
        float m_value;
        std::vector<float> m_prior;
        
        Node(std::shared_ptr<base::Game> game) : Node(nullptr, game) {}
        Node(std::shared_ptr<Node> parent, std::unique_ptr<base::Game> game) : 
            Node(parent, (std::shared_ptr<base::Game>) std::move(game)) {}
        Node(std::shared_ptr<Node> parent, std::shared_ptr<base::Game> game) {
            this->m_parent = parent;
            this->m_children = {};
            this->m_game = game;
            this->m_visit = 1;
            this->m_value = 0;
            this->m_prior = {};
        }
    };
    std::vector<std::vector<uint8_t>> get_eval_states
        (std::vector<std::shared_ptr<Node>>&, int);
    void select(std::vector<std::shared_ptr<Node>>&,
                std::vector<std::shared_ptr<Node>>&,
                int, float, float, float, int);
    std::vector<float> expand(std::vector<std::shared_ptr<Node>>&);
    void backprop(std::vector<std::shared_ptr<Node>>&,
                  std::vector<std::shared_ptr<Node>>&,
                  std::vector<float>&, std::vector<float>&);
}

#endif
