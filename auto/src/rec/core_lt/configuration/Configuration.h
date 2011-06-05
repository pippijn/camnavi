//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_CONFIGURATION_CONFIGURATION_H_
#define _REC_CORE_LT_CONFIGURATION_CONFIGURATION_H_

#include "rec/core_lt/defines.h"

#include "rec/core_lt/configuration/Exception.h"

#include <string>
#include <list>
#include <map>

#ifdef WIN32
#pragma warning( push )
#pragma warning( disable : 4251 )
#endif


//TODO make thread safe

namespace ticpp
{
	class Element;
}

namespace rec
{
	namespace core_lt
	{
		namespace configuration
		{
			class REC_CORE_LT_EXPORT Configuration
			{
			public:
				/**
				* Constructor
				*/
				Configuration();

				/**
				* Constructor
				*
				* @param name The name of this element
				*/
				Configuration( const std::string& name );

				/**
				* Copy-Constructor. Does a deep copy
				*/
				Configuration( const Configuration &config );

				/**
				* Destructor
				*/
				virtual ~Configuration();

				/**
				* @return The name of this element
				*/
				std::string getName() const;

				/**
				* Removes all elements and child configurations from this configuration
				*/
				void clear();

				/**
				* Adds a new string element to the configuration.
				* If the configuration already contains an element with the given name, the old element will be overriden
				*
				* @param key The name of the element
				* @param value The value of the element
				*/
				void put( const std::string& key, const std::string& value );

				/**
				* Adds a new element to the configuration.
				* If the configuration already contains an element with the given name, the old element will be overriden
				*
				* @param key The name of the element
				* @param value The value of the element
				* @throws Exception If stringstream is unable to convert the value to the requested type
				*/
				template <typename T> void put( const std::string& key, T value );

				/**
				*	Retrieves an element from the configuration.
				*
				*	@param key The name of the element
				*  @return The element
				*  @throws Exception If the configuration doesn't contain an element with the given name 
				*/
				std::string get( const std::string& key ) const;

				/**
				*	Retrieves an element from the configuration.
				*
				*	@param key The name of the element
				*  @return The element
				*  @throws Exception If the configuration doesn't contain an element with the given name 
				*/
				template <typename T> T get( const std::string& key ) const;

				/**
				*	Retrieves an element from the configuration.
				*
				*	@param key The name of the element
				*	@param defaultValue Default value for element
				*  @return The element
				*/
				template <typename T> T getWithDefault( const std::string& key, const T& defaultValue ) const;

				/**
				*	Removes an element from the configuration.
				*
				*	@param key The name of the element
				*  @throws Exception If the configuration doesn't contain an element with the given name 
				*/
				void remove( const std::string& key );

				Configuration& addChild( const Configuration &config );

				Configuration& getFirstChild( const std::string& name ) const;

				void getChildren( const std::string& name, std::list<Configuration*>* childList ) const;

				void getChildren( std::list<Configuration*>* childList ) const;

				/**
				*	Saves the configuration to a file
				*
				*	@param xmlFileName The path of the file
				*  @throws Exception If an XML error occurs
				*/
				void save( const std::string& xmlFileName ) const;

				/**
				*	Loads the configuration from a file.
				*  All previous elements of this configuration are removed.
				*
				*	@param xmlFileName The path of the file
				*  @throws Exception If an XML error occurs
				*/
				void load( const std::string& xmlFileName );

			private:
				void save( ticpp::Element *element ) const;
				void load( ticpp::Element *element );

				std::string _name;
				std::multimap< std::string, Configuration* > _children;
				std::map< std::string, std::string > _elements;
			};

			template <typename T> void Configuration::put( const std::string& key, T value )
			{
				std::ostringstream os;
				os << value;
				_elements[ key ] = os.str();
			}

			template <typename T> T Configuration::get( const std::string& key ) const
			{
				try
				{
					T value;
					std::istringstream is( get( key ) );
					is >> value;
					return value;
				}
				catch( Exception &e )
				{
					throw e;
				}
				catch( std::exception )
				{
					throw Exception::createCannotConvertException( this->getName(), key );
				}
			}

			template <typename T> T Configuration::getWithDefault( const std::string& key, const T& defaultValue ) const
			{
				try
				{
					T value;
					std::istringstream is( get( key ) );
					is >> value;
					return value;
				}
				catch( std::exception& )
				{
				}
				return defaultValue;
			}
		}
	}
}

#ifdef WIN32
#pragma warning( pop )
#endif

#endif
