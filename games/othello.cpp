#include <algorithm>
#include <sstream>
#include "othello.hpp"

template<typename ... Args>
std::string string_format(const std::string& format, Args ... args) {
    size_t size = snprintf(nullptr, 0, format.c_str(), args ...) + 1;
    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, format.c_str(), args ...);
    return std::string(buf.get(), buf.get() + size - 1);
}

namespace othello {

    typedef uint64_t (*shift)(uint64_t);
    
    uint64_t up(uint64_t board) {
        return (board >> 8) & 0x00FFFFFFFFFFFFFFLL;
    }
    uint64_t down(uint64_t board) {
        return (board << 8) & 0xFFFFFFFFFFFFFF00LL;
    }
    uint64_t left(uint64_t board) {
        return (board >> 1) & 0x7F7F7F7F7F7F7F7FLL;
    }
    uint64_t right(uint64_t board) {
        return (board << 1) & 0xFEFEFEFEFEFEFEFELL;
    }
    uint64_t up_left(uint64_t board) {
        return (board >> 9) & 0x007F7F7F7F7F7F7FLL;
    }
    uint64_t up_right(uint64_t board) {
        return (board >> 7) & 0x00FEFEFEFEFEFEFELL;
    }
    uint64_t down_left(uint64_t board) {
        return (board << 7) & 0x7F7F7F7F7F7F7F00LL;
    }
    uint64_t down_right(uint64_t board) {
        return (board << 9) & 0xFEFEFEFEFEFEFE00LL;
    }
    
    shift shifts[8] = {&up, &down, &left, &right,
        &up_left, &up_right, &down_left, &down_right};
    
    
    Game::Game() : Game(1) {}
        
    Game::Game(uint8_t turn) {
        if (turn == 1) {
            this->m_b_board = 0x0000001008000000LL;
            this->m_w_board = 0x0000000810000000LL;
            this->m_turn = 1;
            this->m_pass_count = 0;
        } else {
            this->m_turn = turn;
        }
        this->m_actions = {65};
        this->m_state_init = 0;
    }

    int Game::board_height() { return  8; }
    int Game::board_width()  { return  8; }
    int Game::ac_dim()       { return 65; }
    int Game::state_depth()  { return  2; }
	int Game::const_depth()  { return  1; }
    
    std::vector<uint8_t> Game::get_state() {
        if (this->m_state_init) {
            return this->m_state;
        }
        
        std::vector<uint8_t> state(2 * 64);
        uint64_t b_board = this->m_b_board;
        uint64_t occupied = b_board | this->m_w_board;
        int i = 63;
        while (i >= 0) {
            int trailing_zeros = __builtin_ctzll(occupied);
            if (i < trailing_zeros) {
                break;
            }
            i -= trailing_zeros;
            b_board  >>= trailing_zeros;
            occupied >>= trailing_zeros;
            if (b_board & 1) {
                state[i] = 1;
            } else {
                state[i + 64] = 1;
            }
            --i;
            b_board  >>= 1;
            occupied >>= 1;
        }
        this->m_state = state;
        this->m_state_init = 1;
        return state;
    }
	
	std::vector<uint8_t> Game::get_const() {
		std::vector<uint8_t> const_vec(64, this->m_turn & 1);
		return const_vec;
	}
    
    // reference: https://gist.github.com/davidrobles/4042418
    std::vector<uint16_t> Game::get_action() {
        if (this->m_actions[0] != 65) {
            return this->m_actions;
        }
        
        uint64_t curr = this->m_turn & 1 ? this->m_b_board : this->m_w_board;
        uint64_t oppo = this->m_turn & 1 ? this->m_w_board : this->m_b_board;
        uint64_t empty = ~(curr | oppo);
        uint64_t actions = 0LL;
        
        for (auto & s : shifts) {
            uint64_t moves = s(curr) & oppo;
            while (moves) {
                moves = s(moves);
                actions |= moves & empty;
                moves &= oppo;
            }
        }
        
        int count = __builtin_popcountll(actions);
        std::vector<uint16_t> idx(count ? count : 1);
        if (count) {
            int i = 0;
            while (count) {
                int trailing_zeros = __builtin_ctzll(actions);
                i += trailing_zeros;
                idx[--count] = i;
                ++i;
                actions >>= (trailing_zeros + 1);
            }
        } else {
            idx[0] = 64;
        }
        this->m_actions = idx;
        return idx;
    }
    
    // reference: https://gist.github.com/davidrobles/4042418
    std::unique_ptr<base::Game> Game::take_action(uint16_t idx) {
        auto new_game = std::make_unique<Game>(this->m_turn + 1);
        if (idx == 64) {
            new_game->m_b_board = this->m_b_board;
            new_game->m_w_board = this->m_w_board;
            new_game->m_pass_count = this->m_pass_count + 1;
        } else {
            uint64_t curr = this->m_turn & 1 ? this->m_b_board : this->m_w_board;
            uint64_t oppo = this->m_turn & 1 ? this->m_w_board : this->m_b_board;
            uint64_t move = 1LL << idx;
            curr |= move;
            uint64_t all_cell[64];
            int all_count = 0;
            
            for (auto & s : shifts) {
                uint64_t last = 0;
                uint64_t next = s(move) & oppo;
                uint64_t tmp_cell[64];
                int tmp_count = 0;
                while (next) {
                    tmp_cell[tmp_count++] = next;
                    next = s(next);
                    last = next & curr;
                    next &= oppo;
                }
                if (last) {
                    for (int i = 0; i < tmp_count; ++i) {
                        all_cell[all_count++] = tmp_cell[i];
                    }
                }
            }
            for (int i = 0; i < all_count; ++i) {
                curr |= all_cell[i];
                oppo &= ~all_cell[i];
            }
            new_game->m_b_board = this->m_turn & 1 ? curr : oppo;
            new_game->m_w_board = this->m_turn & 1 ? oppo : curr;
            new_game->m_pass_count = 0;
        }
        return new_game;
    }
    
    int Game::terminal() {
        return this->m_pass_count < 2;
    }
    
    int Game::evaluate() {
        int diff = __builtin_popcountll(this->m_b_board) -
            __builtin_popcountll(this->m_w_board);
        return diff ? (diff > 0 ? 1 : -1) : 0;
    }

    std::string Game::state_str() {
        std::vector<uint8_t> state = this->get_state();
        std::stringstream ss;
        for (size_t i = 0; i < 64; ++i) {
            if ((i & 0b111) == 0) {
                ss << '[';
            }
            if (state[i]) {
                ss << 'o';
            } else if (state[i + 64]) {
                ss << 'x';
            } else {
                ss << '.';
            }
            if ((i & 0b111) == 7) {
                ss << "]\n";
            } else {
                ss << ' ';
            }
        }
        return ss.str();
    }

    std::vector<std::string> Game::action_str() {
        std::vector<uint16_t> actions = this->get_action();
        std::vector<std::string> a_str(actions.size());
        std::transform(actions.begin(), actions.end(), a_str.begin(),
            [&](uint16_t idx) { return idx == 64 ? "pass" :
                string_format("(%d, %d)", (63 - idx) >> 3, (63 - idx) & 0b111);
        });
        return a_str;
    }
}
