//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_CORE_LT_CONFIGURATION_SIMPLECONFIG_H_
#define _REC_CORE_LT_CONFIGURATION_SIMPLECONFIG_H_

#include <boost/bind.hpp>
#include <boost/noncopyable.hpp>
#include <boost/function.hpp>
#include <boost/thread/mutex.hpp>
#include "rec/core_lt/configuration/Configuration.h"
#include "rec/core_lt/utils.h"
#include <list>
#include <map>
#include <string>

/*
  howto use:
  parse a simple configuration file like this:
<?xml version="1.0" ?>
<Config>
  <Port value="127.0.0.1" />
  <Settings a="1" b="2" />
</Config>

  first, declare a class:

  class Config : public rec::core_lt::configuration::SimpleConfig
  {
  public:
    static Config& sgt()
    {
      static Config c;
      return c;
    }

  private:
    void myValue( Configuration* c )
    {
      // c wraps around the "Settings" tag
      _values[ "S" ] = "1234";
    }

    Config()
      : SimpleConfig()
    {
      REGISTER_HANDLER_VALUE( "Port", "7626" );
      REGISTER_HANDLER_CUSTOM( "Settings", Config::myValue );
    }
  };

  
  then, initialize it with the filename:
  Config::sgt().initialize( "myfile.xml" );

  then access your values:
  int p = Config::sgt().getInt( "Port" );
     or
  std::string str = Config::sgt().get( "Port" );
*/

#define REGISTER_HANDLER_VALUE( Key, DefaultValue ) \
  _configHandler[ Key ] = boost::bind( &rec::core_lt::configuration::SimpleConfig::hValue, this, _1 );\
  _values[ Key ] = DefaultValue;

#define REGISTER_HANDLER_CUSTOM( Key, FunctionName ) \
  _configHandler[ Key ] = boost::bind( &FunctionName, this, _1 );

namespace rec
{
  namespace core_lt
  {
    namespace configuration
    {
      // thread safe, usage of simple configuration files
      class SimpleConfig : boost::noncopyable
      {
      public:
        SimpleConfig()
          : _mainConf()
          ,_conf( _mainConf )
        {
        }

        void initialize( const std::string& filename )
        {
          _mainConf.load( filename );
          _conf = _mainConf.getFirstChild( "Config" );
        }

        std::string get( const std::string& key ) const
        {
          boost::mutex::scoped_lock l( _configMutex );
          std::map< std::string, std::string >::const_iterator ci = _values.find( key );
          if( _values.end() != ci )
          {
            return ci->second;
          }
          else
          {
            return std::string();
          }
        }

        int getInt( const std::string& key ) const
        {
          boost::mutex::scoped_lock l( _configMutex );
          std::map< std::string, std::string >::const_iterator ci = _values.find( key );
          if( _values.end() != ci )
          {
            int v;
            std::istringstream is( ci->second );
            is >> v;
            return v;
          }
          else
          {
            return 0;
          }
        }

        void set( const std::string& key, const std::string& value )
        {
          boost::mutex::scoped_lock l( _configMutex );
          _values[ key ] = value;
        }

        void setInt( const std::string& key, int value )
        {
          boost::mutex::scoped_lock l( _configMutex );
          _values[ key ] = rec::core_lt::toString< int >( value );
        }

        // must be public for boost::bind :-(
        void hValue( Configuration* c )
        {
          _values[ c->getName() ] = c->get( "value" );
        }

      protected:
        // has to be called in client constructor after setting handler functions
        void extract()
        {
          std::list< Configuration* > childList;
          _conf.getChildren( &childList );
          std::list< Configuration* >::const_iterator ci = childList.begin();
          while( ci != childList.end() )
          {
            std::map< std::string, boost::function< void( Configuration* ) > >::const_iterator mapIter = _configHandler.find( (*ci)->getName() );
            if( mapIter != _configHandler.end() )
            {
              // supported value
              (mapIter->second)( *ci );
            }
            else
            {
              throw rec::core_lt::Exception( (*ci)->getName() + " not supported" );
            }
            ci++;
          }
        }

      private:
        // save configuration
        Configuration _mainConf;
        mutable boost::mutex _configMutex;

      protected:
        std::map< std::string, boost::function< void( Configuration* ) > > _configHandler;
        std::map< std::string, std::string > _values;
        Configuration& _conf;

      };
    }
  }
}

#endif
