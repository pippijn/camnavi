//  Copyright (C) 2004-2009, Robotics Equipment Corporation GmbH

#include "rec/core_lt/configuration/Configuration.h"

#include "ticpp.h"

#include <sstream>

using rec::core_lt::configuration::Configuration;
using rec::core_lt::configuration::Exception;

Configuration::Configuration()
: _name( "Configuration" )
{
}

Configuration::Configuration( const std::string& name )
: _name( name )
{
}

Configuration::Configuration( const Configuration& config )
: _name( config._name )
  ,_elements( config._elements )
{
  //Do a deep copy of the child configurations
  std::multimap< std::string, Configuration* >::const_iterator iter = config._children.begin();
  while(config._children.end() != iter)
  {
    Configuration *child = new Configuration( *(iter->second) );
    _children.insert( std::make_pair<std::string, Configuration*>( child->getName(), child ) );

    iter++;
  }
}

Configuration::~Configuration( )
{
  std::multimap< std::string, Configuration* >::const_iterator iter = _children.begin();
  while(_children.end() != iter)
  {
    delete iter->second;
    iter++;
  }
}

std::string Configuration::getName() const
{
  return _name;
}

void Configuration::clear()
{
  _elements.clear();

  std::multimap< std::string, Configuration* >::const_iterator iter = _children.begin();
  while(_children.end() != iter)
  {
    delete iter->second;
    iter++;
  }
  _children.clear();
}

void Configuration::put( const std::string& key, const std::string& value )
{
  _elements[ key ] = value;
}

std::string Configuration::get( const std::string& key ) const
{
  std::map< std::string, std::string >::const_iterator iter = _elements.find( key );
  if(_elements.end() != iter)
  {
    return (*iter).second;
  }

  throw Exception::createElementNoFoundException( this->getName(), key );
}

void Configuration::remove( const std::string& key )
{
  std::map< std::string, std::string >::iterator iter = _elements.find( key );
  if(_elements.end() != iter)
  {
    _elements.erase( iter );
  }
  else
  {
    throw Exception::createElementNoFoundException( this->getName(), key );
  }
}

Configuration& Configuration::addChild( const Configuration& config )
{
  Configuration* newChild = new Configuration( config );

  _children.insert( std::make_pair<std::string, Configuration*>( config.getName(), newChild) );

  return *newChild;
}

Configuration& Configuration::getFirstChild( const std::string& name ) const
{
  std::multimap< std::string, Configuration* >::const_iterator iter = _children.find( name );
  if(_children.end() != iter)
  {
    return *(iter->second);
  }

  throw Exception::createElementNoFoundException( this->getName(), name );
}

void Configuration::getChildren( const std::string& name, std::list<Configuration*>* childList) const
{
  std::pair< std::multimap< std::string, Configuration*>::const_iterator,
             std::multimap< std::string, Configuration*>::const_iterator > range = _children.equal_range( name );

  std::multimap< std::string, Configuration* >::const_iterator iter = range.first;
  while(range.second != iter)
  {
    childList->push_back( iter->second );
    iter++;
  }
}

void Configuration::getChildren( std::list<Configuration*>* childList ) const
{
  std::multimap< std::string, Configuration* >::const_iterator iter = _children.begin();
  while(_children.end() != iter)
  {
    childList->push_back( iter->second );
    iter++;
  }
}

void Configuration::save( const std::string& xmlFileName ) const
{
  try
  {
    //Create the document
    ticpp::Document doc;  
 	  ticpp::Declaration* decl = new ticpp::Declaration( "1.0", "", "" );  
	  doc.LinkEndChild( decl );

    //Add the root element
    ticpp::Element *root = new ticpp::Element( _name );  
	  doc.LinkEndChild( root );  

    //Create the child elements
    save( root );

    //Save to file
    doc.SaveFile( xmlFileName );
  }
  catch( ticpp::Exception &e )
  {
    throw Exception( e.m_details );
  }
}

void Configuration::save( ticpp::Element *element ) const
{
  //Serialize configuration elements as xml attributes
  std::map< std::string, std::string >::const_iterator elementIter = _elements.begin();
  while(_elements.end() != elementIter)
  {
    element->SetAttribute( elementIter->first, elementIter->second );
    elementIter++;
  }

  //Serialize child configurations as xml elements
  std::multimap< std::string, Configuration* >::const_iterator childIter = _children.begin();
  while(_children.end() != childIter)
  {
    std::string childName = childIter->first;
    Configuration* childConfig = childIter->second;

    ticpp::Element *childElement = new ticpp::Element( childName );
    element->LinkEndChild( childElement );

    childConfig->save( childElement );
    
    childIter++;
  }
}

void Configuration::load( const std::string& xmlFileName )
{
  try
  {
    clear();

    //Open the document
    ticpp::Document doc( xmlFileName );
    doc.LoadFile();

    //Read the top element
    ticpp::Element* topElem = doc.FirstChildElement();
    _name = topElem->Value();

    //Read the child elements
    load( topElem );
  }
  catch( ticpp::Exception &e )
  {
    throw Exception( e.m_details );
  }
}

void Configuration::load( ticpp::Element *element )
{
  for( ticpp::Iterator<ticpp::Attribute> i = element->FirstAttribute(false); i != NULL && i != i.end(); i++ )
  {
    put( i->Name(), i->Value() );
  }

  for( ticpp::Iterator<ticpp::Element> i = element->FirstChildElement(false); i != NULL && i != i.end(); i++ )
  {
    std::string childName = i->Value();
    Configuration *childConfig = new Configuration( childName );
    _children.insert( std::make_pair<std::string, Configuration*>( childName, childConfig ) );

    childConfig->load( i->ToElement() );
  }
}
