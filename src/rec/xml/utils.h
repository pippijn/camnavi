//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#ifndef _REC_XML_UTILS_H_
#define _REC_XML_UTILS_H_

#include <sstream>
#include "rec/xml/ticpp.h"
#include "rec/core_lt/Exception.h"
#include <string>

namespace rec
{
  namespace xml
  {
    namespace utils
    {
      // takes care of deletion of el
      static std::string toString( TiXmlElement* el )
      {
        TiXmlDocument doc;
        TiXmlDeclaration* decl = new TiXmlDeclaration( "1.0", "", "" );
        doc.LinkEndChild( decl );
        doc.LinkEndChild( el );

        std::ostringstream os;
        os << doc;
        return os.str();
      }

      static const TiXmlElement* getChild( const TiXmlElement* el, const char* name )
      {
        const TiXmlElement* res = el->FirstChildElement( name );
        if( NULL == res )
        {
          throw rec::core_lt::Exception( std::string("No element \"") + name + ("\"") );
        }
        return res;
      }

      template <typename T> 
      static void getAttributeDefault( const TiXmlElement* element, const char* name, T* value )
      {
        const char* attrib = element->Attribute( name );
        if( NULL != attrib )
        {
          std::istringstream is( attrib );
          is >> *value;
        }
      }

      template <typename T> 
      static void getAttributeThrow( const TiXmlElement* element, const char* name, T* value )
      {
        const char* attrib = element->Attribute( name );
        if( NULL != attrib )
        {
          std::istringstream is( attrib );
          is >> *value;
        }
        else
        {
          throw rec::core_lt::Exception( std::string("No attribute \"") + name + ("\" in element") + std::string( element->Value() ) );
        }
      }

      static void getAttributeThrow( const TiXmlElement* element, const char* name, std::string* value )
      {
        const char* attrib = element->Attribute( name );
        if( NULL != attrib )
        {
          *value = attrib;
        }
        else
        {
          throw rec::core_lt::Exception( std::string("No attribute \"") + name + ("\" in element") + std::string( element->Value() ) );
        }
      }

    }
  }
}

#endif
