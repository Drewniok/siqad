// @file:     electrode.h
// @author:   Nathan
// @created:  2017.10.27
// @editted:  2017.10.27 - Nathan
// @license:  GNU LGPL v3
//
// @desc:     Electrode objects for creation, deletion, and moving of electrodes.

#ifndef _GUI_PR_ELECTRODE_H_
#define _GUI_PR_ELECTRODE_H_

#include <QtWidgets>
#include "item.h"

namespace prim{

  // forard declarations
  class Layer;

  class Electrode: public prim::Item
  {
  public:

    // constructor, create an electrode given two points
    Electrode(int lay_id, QPoint p1, QPoint p2);
    // destructor
    ~Electrode(){}

    // accessors

    QPoint getp1(void){ return p1;}
    QPoint getp2(void){ return p2;}
    QPoint gettopLeft(void) {return topLeft;}
    // inherited abstract method implementations

    QRectF boundingRect() const Q_DECL_OVERRIDE;
    void paint(QPainter *, const QStyleOptionGraphicsItem *, QWidget *) Q_DECL_OVERRIDE;

    Item *deepCopy() const;
  //
  protected:
    virtual void mousePressEvent(QGraphicsSceneMouseEvent *e) Q_DECL_OVERRIDE;

  private:

    // construct static variables
    void constructStatics();

    // VARIABLES
    QPoint p1;
    QPoint p2;
    QPoint topLeft;

    int elec_width;
    int elec_height;

    static qreal edge_width;  // proportional width of dot boundary edge

    // static QColor edge_col;   // boundary edge color
    static QColor fill_col;   // dot fill color (same for all lattice dots)

    static qreal in_fill;       // fil factor for inner circle
    static QColor in_fill_col;  // colour of inner circle

    static QColor edge_col;     // edge colour, unselected
    static QColor selected_col; // edge colour, selected
  };

} // end prim namespace

#endif
