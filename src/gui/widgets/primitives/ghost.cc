// @file:     ghost.cc
// @author:   Jake
// @created:  2016.11.24
// @editted:  2017.06.07  - Jake
// @license:  GNU LGPL v3
//
// @desc:     Implementation of GhostDot and Ghost


#include "ghost.h"
#include "dbdot.h"


// GHOSTDOT CLASS
qreal prim::GhostDot::diameter = -1;

prim::GhostDot::GhostDot(prim::Item *item, prim::Item *parent, QColor *pcol)
 : Item(prim::Item::GhostDot, 0, parent), pcol(pcol)
{
  //settings::GUISettings *gui_settings = settings::GUISettings::instance();

  if(diameter<0)
    constructStatics();

  // create dot at item center, assumes item local boundingRect centered at 0.
  setPos(item->pos());

  // get the DB's lattice coordinate
  lat_coord = static_cast<prim::DBDot*>(item)->latticeCoord();
}

QRectF prim::GhostDot::boundingRect() const
{
  return QRectF(-.5*diameter, -.5*diameter, diameter, diameter);
}

void prim::GhostDot::paint(QPainter *painter, const QStyleOptionGraphicsItem*, QWidget*)
{
  painter->setPen(Qt::NoPen);
  painter->setBrush(*pcol);
  painter->drawEllipse(boundingRect());
}

void prim::GhostDot::constructStatics()
{
  settings::GUISettings *gui_settings = settings::GUISettings::instance();
  diameter = gui_settings->get<qreal>("ghost/dot_diameter")*scale_factor;
}



//GHOSTBOX CLASS
prim::GhostBox::GhostBox(prim::Item *item, prim::Item *parent)
  : Item(prim::Item::GhostBox, 0, parent)
{
  constructStatics();

  if (item->item_type == prim::Item::Electrode) {
    width = static_cast<prim::Electrode*>(item)->getWidth();
    height = static_cast<prim::Electrode*>(item)->getHeight();
  } else if (item->item_type == prim::Item::AFMArea) {
    prim::AFMArea *afm_area = static_cast<prim::AFMArea*>(item);
    width = afm_area->bottomRight().x()-afm_area->topLeft().x();
    height = afm_area->bottomRight().y()-afm_area->topLeft().y();
  } else {
    qFatal("Trying to make a GhostBox out of unsupported item type");
  }

  setZValue(-1);
  setPos(item->pos());
}

QRectF prim::GhostBox::boundingRect() const
{
  return QRectF(0, 0, width, height);
}

void prim::GhostBox::paint(QPainter *painter, const QStyleOptionGraphicsItem*, QWidget*)
{
  painter->setPen(Qt::NoPen);
  painter->setBrush(ghost_box_color);
  painter->drawRect(boundingRect());
}

void prim::GhostBox::constructStatics()
{
  settings::GUISettings *gui_settings = settings::GUISettings::instance();
  ghost_box_color = gui_settings->get<QColor>("ghostbox/valid_col");
}


// GHOST CLASS
prim::Ghost* prim::Ghost::inst = 0;

prim::Ghost* prim::Ghost::instance()
{
  if(inst==0)
    inst = new prim::Ghost();
  return inst;
}


void prim::Ghost::cleanGhost()
{
  // clear lists
  sources.clear();
  for(prim::GhostDot *dot : dots)
    delete dot;
  dots.clear();

  // qDebug() << QObject::tr("Deleting Ghost Box");
  box_sources.clear();
  for(prim::GhostBox *box : boxes)
    delete box;
  boxes.clear();

  setPos(0,0);
  setValid(true);

  anchor=0;
  valid_hash.clear();

  aggnode.reset();

  hide();
}


void prim::Ghost::prepare(const QList<prim::Item*> &items, QPointF scene_pos)
{
  cleanGhost();
  for(prim::Item *item : items)
    prepareItem(item, &aggnode);
  zeroGhost(scene_pos);
  setAnchor();
  show();
}


void prim::Ghost::moveTo(QPointF pos)
{
  setPos(pos-zero_offset);
}


void prim::Ghost::moveByCoord(prim::LatticeCoord coord_offset, prim::Lattice *lattice)
{
  prim::LatticeCoord fin_coord = anchor->latticeCoord() + coord_offset;
  qDebug() << QObject::tr("anchor (%1,%2,%3) + offset (%4,%5,%6) = final (%7,%8,%9)").arg(anchor->latticeCoord().n).arg(anchor->latticeCoord().m).arg(anchor->latticeCoord().l).arg(coord_offset.n).arg(coord_offset.m).arg(coord_offset.l).arg(fin_coord.n).arg(fin_coord.m).arg(fin_coord.l);

  for (prim::GhostDot *dot : dots)
    dot->setLatticeCoord(dot->latticeCoord() + coord_offset);
  QPointF offset = lattice->latticeCoord2ScenePos(coord_offset);
  instance()->moveBy(offset.x(), offset.y());
}


QList<prim::Item*> prim::Ghost::getTopItems() const
{
  // each top item corresponds to one of the top level nodes in aggnode

  // qDebug() << QObject::tr("Ghost::getTopItems");
  QList<prim::Item*> items;
  for(prim::AggNode *node : aggnode.nodes)
    items.append(getNodeItem(node));
  return items;
}


QList<bool> prim::Ghost::getLatticeAvailability(const prim::LatticeCoord &offset,
    prim::Lattice *lattice) const
{
  QList<bool> avail;
  for (int i=0; i<dots.count(); i++) {
    qDebug() << QObject::tr("ghost dot at (%1, %2, %3)").arg(dots.at(i)->latticeCoord().n).arg(dots.at(i)->latticeCoord().m).arg(dots.at(i)->latticeCoord().l);
    if (!lattice->isValid(dots.at(i)->latticeCoord()+offset)) {
      qDebug() << "Target site is not valid";
      avail.append(false);
      continue;
    }
    if (lattice->isOccupied(dots.at(i)->latticeCoord()+offset)) {
      qDebug() << "Target site is occupied";
      avail.append(false);
      continue;
    }
    avail.append(true);
  }
  return avail;
}


prim::LatticeCoord prim::Ghost::getLatticeCoord(prim::DBDot *db) const
{
  // get index of source
  int index = sources.indexOf(static_cast<prim::Item*>(db));
  if (index==-1)
    return prim::LatticeCoord(0,0,-1);
  else
    return dots.at(index)->latticeCoord();
}

QPointF prim::Ghost::freeAnchor(QPointF scene_pos)
{
  return scene_pos+anchor_offset;
}


void prim::Ghost::setValid(bool val)
{
  settings::GUISettings *gui_settings = settings::GUISettings::instance();

  if(valid != val){
    valid = val;
    if(valid)
      col = gui_settings->get<QColor>("ghost/valid_col");
    else
      col = gui_settings->get<QColor>("ghost/invalid_col");
  }
}


bool prim::Ghost::checkValid(const prim::LatticeCoord &offset, prim::Lattice *lattice)
{
  QList<bool> lattice_avail = getLatticeAvailability(offset, lattice);

  // invalid if a dangling bond is associated with no selectable lattice dot or
  // an unselectable lattice dot
  for(int i=0; i<dots.count(); i++)
    if (!lattice_avail.at(i))
      return false;

  return true;
}


QPointF prim::Ghost::moveOffset() const
{
  if(sources.count())
    return dots.first()->scenePos()-sources.first()->scenePos();
  else
    return QPointF();
}


QRectF prim::Ghost::boundingRect() const
{
  return QRectF();
}

void prim::Ghost::paint(QPainter*, const QStyleOptionGraphicsItem*, QWidget*)
{}



void prim::Ghost::echoTopIndices()
{
  QString s;
  echoNode(s, &aggnode);
  qDebug() << s;
}

void prim::Ghost::echoNode(QString &s, prim::AggNode *node)
{
  qDebug() << QObject::tr("Entering node: ind=%1 :: count=%2").arg(node->index).arg((node->nodes).count());
  if(node->index<0){
    // new Aggregate
    s += '[';
    for(prim::AggNode *n : node->nodes)
      echoNode(s, n);
    s += ']';
  }
  else
    s += QString("%1,").arg(node->index);
}


// PRIVATE METHODS


prim::Ghost::Ghost()
 : Item(prim::Item::Ghost)
{
  valid=false;
  cleanGhost();
  setVisible(false);
}

void prim::Ghost::createGhostDot(prim::Item *item)
{
  if(item->item_type == prim::Item::Aggregate)
    qWarning() << QObject::tr("Creating a ghost dot for an aggregate...");

  prim::GhostDot *dot = new prim::GhostDot(item, this, &col);

  dots.append(dot);
  sources.append(item);
}


void prim::Ghost::createGhostBox(prim::Item *item)
{
  qDebug() << QObject::tr("Creating Ghost Box");
  // prim::GhostDot *dot = new prim::GhostDot(item, this, &col);
  prim::GhostBox *box = new prim::GhostBox(item, this);
  //
  boxes.append(box);
  box_sources.append(item);
}


void prim::Ghost::prepareItem(prim::Item *item, prim::AggNode *node)
{
  prim::AggNode *new_node;
  if (item->item_type == prim::Item::Aggregate) {
    // add a new list-type IndexList
    new_node = new prim::AggNode();
    new_node->source_type = prim::AggNode::Aggregate;
    node->nodes.append(new_node);
    // add each item in the Aggregate to the new list
    prim::Aggregate *agg = static_cast<prim::Aggregate*>(item);
    for(prim::Item *it : agg->getChildren())
      prepareItem(it, new_node);
  } else if (item->item_type == prim::Item::DBDot) {
    // add a new index-type IndexList
    new_node = new prim::AggNode(sources.count());
    new_node->source_type = prim::AggNode::DBDot;
    node->nodes.append(new_node);
    // create a GhostDot for the Item
    createGhostDot(item);
  } else if (item->item_type == prim::Item::Electrode) {
    // add a new index-type IndexList
    new_node = new prim::AggNode(box_sources.count());
    new_node->source_type = prim::AggNode::Electrode;
    node->nodes.append(new_node);
    // create a GhostBox for the Item
    createGhostBox(item);
  } else if (item->item_type == prim::Item::AFMArea) {
    new_node = new prim::AggNode(box_sources.count());
    new_node->source_type = prim::AggNode::AFMArea;
    node->nodes.append(new_node);
    createGhostBox(item);
  }
}


void prim::Ghost::updateValid()
{

}


void prim::Ghost::zeroGhost(QPointF scene_pos)
{
  if (!scene_pos.isNull()) {
    zero_offset = scene_pos;
    return;
  }
  // compute center of effective bounding rect for all items
  qreal xmin=0, xmax=0, ymin=0, ymax=0;
  bool unset=true;
  for(prim::GhostDot *dot : dots){
    QRectF rect = dot->boundingRect();
    rect.moveCenter(dot->pos());
    if(unset){
      unset=false;
      xmin = rect.left();
      xmax = rect.right();
      ymin = rect.top();
      ymax = rect.bottom();
    }
    else{
      xmin = qMin(xmin, rect.left());
      xmax = qMax(xmax, rect.right());
      ymin = qMin(ymin, rect.top());
      ymax = qMax(ymax, rect.bottom());
    }
  }

  zero_offset = QPointF(.5*(xmin+xmax), .5*(ymin+ymax));
}


void prim::Ghost::setAnchor()
{
  // find nearest GhostDot to the zero_offset, by Manhattan length
  prim::GhostDot *dot=anchor=0;
  qreal mdist=-1, dist;
  for(int i=0; i<dots.count(); i++){
    if(sources.at(i)->item_type != prim::Item::DBDot)
      continue;
    dot = dots.at(i);
    dist = (dot->scenePos()-zero_offset).manhattanLength();
    if(mdist < 0 || mdist > dist){
      anchor = dot;
      mdist=dist;
    }
  }

  anchor_offset = anchor==0 ? QPointF() : anchor->scenePos() - zero_offset - pos();
}


prim::Item *prim::Ghost::getNodeItem(prim::AggNode *node) const
{

  // qDebug() << QObject::tr("node index: %1").arg(node->index);
  if(node->index<0){
    // node corresponds to an Aggregate, the parent of the first item in the Aggregate
    return static_cast<prim::Item*>(getNodeItem(node->nodes.first())->parentItem());
  }
  else if(node->source_type == prim::AggNode::DBDot)
    return sources.at(node->index);
  else if(node->source_type == prim::AggNode::Electrode ||
          node->source_type == prim::AggNode::AFMArea)
    return box_sources.at(node->index);
  else
    return 0;

}
