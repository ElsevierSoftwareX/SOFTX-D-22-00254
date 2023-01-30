/********************************************************************************
  Copyright (C) 2020 - 2022 by the lifex authors.

  This file is part of lifex.

  lifex is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  lifex is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with lifex.  If not, see <http://www.gnu.org/licenses/>.

  This file has been readapted from the corresponding file available at
  https://github.com/Dugy/generic_factory/blob/master/generic_factory.hpp,
  released under compatible license terms (please consult the doc/licenses
  directory for more information).
********************************************************************************/

/**
 * @file
 *
 * @author Pasquale Claudio Africa <pasqualeclaudio.africa@polimi.it>.
 */

#ifndef __LIFEX_GENERIC_FACTORY_HPP_
#define __LIFEX_GENERIC_FACTORY_HPP_

#include "source/lifex.hpp"

#include "source/param_handler.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

namespace lifex
{
  /**
   * @brief Generic self-registering factory class.
   *
   * References:
   * -
   * https://lordsoftech.com/programming/generic-self-registering-factory-no-more-need-to-write-factories/
   * - https://github.com/Dugy/generic_factory
   *
   * A macro that hides the ugly part of registering children and that can be
   * used both in header and source files is also defined for convenience.
   * For example, if <kbd>"ChildName"</kbd> is the name of a child of class
   * <kbd>Child</kbd> derived from <kbd>Base</kbd> and constructed from
   * <kbd>const float &</kbd> and <kbd>const unsigned int &</kbd> input
   * arguments:
   * @code{.cpp}
   * LIFEX_REGISTER_CHILD_INTO_FACTORY(Base,
   *                                   Child,
   *                                   "ChildName", // Or Child::label, if any.
   *                                   const float &,
   *                                   const unsigned int &);
   * @endcode
   *
   * For a more advanced use case, please refer to
   * <kbd>LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM</kbd>.
   */
  template <class Parent, class... Args>
  class GenericFactory
  {
  public:
    /**
     * Register a constructor of a possible child. Thread safe.
     *
     * @param[in] name    The name of the child type.
     * @param[in] builder A function that returns a <kbd>std::unique_ptr</kbd>
                          to a newly constructed child when called.
     * @return <kbd>true</kbd> if successfully added,
               <kbd>false</kbd> if the child already exists.
     */
    static bool
    register_child(
      const std::string &                                    name,
      const std::function<std::unique_ptr<Parent>(Args...)> &builder)
    {
      auto &factory = get_generic_factory();

      std::lock_guard<std::mutex> guard(factory.mutex);

      auto found = factory.children.find(name);

      if (found != factory.children.end())
        return false;

      factory.children[name] = builder;

      return true;
    }

    /**
     * Register a child from a constructor.
     * The template argument is the class of the child being created.
     * Thread safe.
     *
     * @param[in] name The name of the child.
     * @return <kbd>true</kbd> if successfully added,
     *         <kbd>false</kbd> if the child already exists.
     */
    template <class Child>
    static bool
    register_child(const std::string &name)
    {
      return register_child(name, [](Args... args) {
        return std::make_unique<Child>(args...);
      });
    }

    /**
     * Unregister a child. Thread safe.
     *
     * @param[in] name The name of the child.
     * @return <kbd>true</kbd> if it was already registered,
     *         <kbd>false</kbd> if it wasn't.
     */
    static bool
    unregister_child(const std::string &name)
    {
      auto &factory = get_generic_factory();

      std::lock_guard<std::mutex> guard(factory.mutex);

      auto found = factory.children.find(name);

      if (found == factory.children.end())
        return false;

      factory.children.erase(found);

      return true;
    }

    /**
     * Create a child given its name and the arguments to be passed to its
     * builder/constructor. Thread safe.
     *
     * @param[in] name The name of the child.
     * @param[in] args Constructor arguments.
     */
    static std::unique_ptr<Parent>
    create_child(const std::string &name, Args... args)
    {
      auto &factory = get_generic_factory();

      std::lock_guard<std::mutex> guard(factory.mutex);

      auto found = factory.children.find(name);

      AssertThrow(found != factory.children.end(),
                  ExcMessage("Unknown child: " + name));

      return found->second(args...);
    }

    /**
     * Get the keys of all registered children (or those matching the
     * subset specified as an input argument). Thread safe.
     */
    static std::set<std::string>
    get_registered_keys(const std::set<std::string> &valid_children = {})
    {
      auto &factory = get_generic_factory();

      std::lock_guard<std::mutex> guard(factory.mutex);

      std::set<std::string> children_names;

      for (const auto &child : factory.children)
        {
          if (valid_children.empty() ||
              valid_children.find(child.first) != valid_children.end())
            children_names.insert(child.first);
        }

      return children_names;
    }

    /**
     * Get the list of keys of all registered children (or those matching the
     * subset specified as an input argument) as a pipe-separated string
     * that can be used to declare <kbd>ParamHandler</kbd> entries. Thread
     * safe.
     */
    static std::string
    get_registered_keys_prm(const std::set<std::string> &valid_children = {})
    {
      auto &factory = get_generic_factory();

      std::lock_guard<std::mutex> guard(factory.mutex);

      std::string children_names_prm;

      for (auto child = factory.children.begin();
           child != factory.children.end();
           ++child)
        {
          if (valid_children.empty() ||
              valid_children.find(child->first) != valid_children.end())
            {
              children_names_prm += child->first;

              if (std::next(child) != factory.children.end())
                children_names_prm += " | ";
            }
        }

      return children_names_prm;
    }

    /**
     * Declare parameters for all registered children.
     */
    static void
    declare_children_parameters(ParamHandler &params, Args... args)
    {
      declare_children_parameters(params, {}, args...);
    }

    /**
     * Same as the function above, declaring also 0D parameters.
     */
    static void
    declare_children_parameters_0d(ParamHandler &params, Args... args)
    {
      declare_children_parameters_0d(params, {}, args...);
    }

    /**
     * Declare parameters for those registered children matching the subset
     * specified as an input argument.
     */
    static void
    declare_children_parameters(ParamHandler &               params,
                                const std::set<std::string> &valid_children,
                                Args... args)
    {
      for (const auto &label : get_registered_keys())
        {
          if (valid_children.empty() ||
              valid_children.find(label) != valid_children.end())
            {
              std::unique_ptr<Parent> child = create_child(label, args...);
              child->declare_parameters(params);
            }
        }
    }

    /**
     * Same as the function above, declaring also 0D parameters.
     */
    static void
    declare_children_parameters_0d(ParamHandler &               params,
                                   const std::set<std::string> &valid_children,
                                   Args... args)
    {
      for (const auto &label : get_registered_keys())
        {
          if (valid_children.empty() ||
              valid_children.find(label) != valid_children.end())
            {
              std::unique_ptr<Parent> child = create_child(label, args...);
              child->declare_parameters(params);
              child->declare_parameters_0d(params);
            }
        }
    }

    /**
     * Create a child given its name and the arguments to be passed to its
     * builder/constructor, parse its parameters and return the newly created
     * istance.
     */
    static std::unique_ptr<Parent>
    parse_child_parameters(ParamHandler &     params,
                           const std::string &label,
                           Args... args)
    {
      std::unique_ptr<Parent> child = create_child(label, args...);
      child->parse_parameters(params);

      return child;
    }

    /**
     * Same as the function above, parsing also 0D parameters.
     */
    static std::unique_ptr<Parent>
    parse_child_parameters_0d(ParamHandler &     params,
                              const std::string &label,
                              Args... args)
    {
      std::unique_ptr<Parent> child = create_child(label, args...);
      child->parse_parameters(params);
      child->parse_parameters_0d(params);

      return child;
    }

  private:
    /// Default constructor (defaulted).
    /// There is no need to forbid copying or moving, because it's impossible to
    /// obtain an instance from outside.
    GenericFactory() = default;

    /**
     * Force the initialization of @ref children by circumventing the
     * "Static Initialization Order Fiasco"
     * (see <a
     * href="https://www.jibbow.com/posts/cpp-header-only-self-registering-types/">here</a>).
     * Since this method is invoked from this very same file, the static member
     * GenericFactory declared inside of it is always initialized before other
     * <kbd>static</kbd> variables that can possibly depend on the @ref children
     * map (@a e.g. those instantiated via the
     * <kbd>LIFEX_REGISTER_CHILD_INTO_FACTORY</kbd> macro).
     */
    static GenericFactory &
    get_generic_factory()
    {
      static GenericFactory factory;
      return factory;
    }

    /// Children registry. It is stored as a map, whose keys are strings and
    /// values are functions returning a pointer to the base class.
    std::map<std::string, std::function<std::unique_ptr<Parent>(Args...)>>
      children;

    /// Mutex lockable object.
    std::mutex mutex;
  };

// Certain compilers are known to optimize away unused static variables,
// which is why the inner boolean variable is declared volatile.
#define LIFEX_REGISTER_CHILD_INTO_FACTORY(INTERFACE_TYPENAME,              \
                                          CHILD_TYPENAME,                  \
                                          CHILD_NAME,                      \
                                          ...)                             \
  namespace GenericFactoryInternals                                        \
  {                                                                        \
    static volatile const bool                                             \
      INTERFACE_TYPENAME##_From_##CHILD_TYPENAME##_Registered =            \
        GenericFactory<INTERFACE_TYPENAME, ##__VA_ARGS__>::register_child< \
          CHILD_TYPENAME>(CHILD_NAME);                                     \
  }                                                                        \
  static_assert(true, "Did you forget a semi-colon after this macro?")


// Provide custom variable name.
#define LIFEX_REGISTER_CHILD_INTO_FACTORY_CUSTOM(                        \
  VAR_NAME, INTERFACE_TYPENAME, CHILD_TYPENAME, CHILD_NAME, ...)         \
  namespace GenericFactoryInternals                                      \
  {                                                                      \
    static volatile const bool VAR_NAME##_Registered =                   \
      GenericFactory<INTERFACE_TYPENAME, ##__VA_ARGS__>::register_child< \
        CHILD_TYPENAME>(CHILD_NAME);                                     \
  }                                                                      \
  static_assert(true, "Did you forget a semi-colon after this macro?")

} // namespace lifex

#endif /* __LIFEX_GENERIC_FACTORY_HPP_ */
