#ifndef QCOLORCOMBOBOX_H
#define QCOLORCOMBOBOX_H

#include <QComboBox>
#include <QToolButton>

class QColorComboBox : public QComboBox
 {
     Q_OBJECT
     Q_PROPERTY(QColor color READ color WRITE setColor USER true)

 public:
             QColorComboBox(QWidget *widget = 0):QComboBox(widget){populateList();}

 public:
     QColor color() const;
     void setColor(QColor c);

 private:
     void populateList();
 };

class QColorToolButton : public QToolButton
 {
     Q_OBJECT
     Q_PROPERTY(QColor color READ color WRITE setColor USER true)

 public:
     QColorToolButton(QWidget *widget = 0):QToolButton(widget)
     {
         setToolButtonStyle(Qt::ToolButtonTextOnly);
     }
     void mouseReleaseEvent(QMouseEvent * e ) override;
 public:
     QColor color() const;
     void setColor(QColor color);
 private:
     QColor selected_color;
 };



#endif // QCOLORCOMBOBOX_H
