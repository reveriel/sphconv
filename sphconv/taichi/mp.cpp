

#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <string>
#include <cassert>
#include <typeinfo>
#include <ratio>
using namespace std;
// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

#include <boost/mp11.hpp>
using namespace boost::mp11;
#include <boost/mpl/advance.hpp>
#include <boost/core/demangle.hpp>


template<typename L,
typename I>
struct add_cons_ {
    using OldI = mp_front<L>;
    static constexpr int new_int = I::value + OldI::value;
    using NewI = std::integral_constant<int, new_int>;

    using type = mp_push_back<L, NewI>;
};

template<typename L,
typename I>
using add_cons = typename add_cons_<L, I>::type;


int main() {
    using L = mp_list<
    std::integral_constant<int, 3>,
     std::integral_constant<int, 4>,
     std::integral_constant<int, 5>
     >;

    std::cout << boost::core::demangle(typeid(mp_identity<L>).name()) << std::endl;

    using Res = mp_fold<L,
    mp_list<std::integral_constant<int, 2>>,
    add_cons>;

    std::cout << boost::core::demangle(typeid(Res()).name()) << std::endl;

    std::cout << boost::core::demangle(typeid(mp_front<Res>()).name()) << std::endl;

    using L1 = mp_list<ratio<1,8> ,ratio<1,4>, ratio<1,2>>;
    using R1 = mp_fold<L1, ratio<0,1>, ratio_add>;
    std::cout << boost::core::demangle(typeid(R1()).name()) << std::endl;


    return 0;
}


