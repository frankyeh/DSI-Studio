#include <QColorDialog>
#include <QPainter>
#include "qcolorcombobox.h"

QColor QColorComboBox::color() const
 {
    return itemData(currentIndex(), Qt::DecorationRole).value<QColor>();
 }

void QColorComboBox::setColor(QColor color)
 {
     setCurrentIndex(findData(color, int(Qt::DecorationRole)));
 }

void QColorComboBox::populateList()
{
    QStringList colorNames = QColor::colorNames();

    for (int i = 0; i < colorNames.size(); ++i) {
        QColor color(colorNames[i]);

        insertItem(i, colorNames[i]);
        setItemData(i, color, Qt::DecorationRole);
    }
}

QColor QColorToolButton::color() const
 {
    return selected_color;
 }
void QColorToolButton::setColor(QColor color)
 {
    selected_color = color;
    QPixmap pixmap(22,22);
    QPainter painter(&pixmap);
    painter.setPen(color);
    painter.setBrush(color);
    painter.drawRect(0,0,22,22);
    QIcon icon;
    icon.addPixmap(pixmap);
    setIcon(icon);
    setToolButtonStyle(Qt::ToolButtonIconOnly);
}
void QColorToolButton::mouseReleaseEvent ( QMouseEvent * e )
{
    setColor(QColorDialog::getColor(selected_color,
                (QWidget*)parent(),"Select color",QColorDialog::ShowAlphaChannel));
    QToolButton::mouseReleaseEvent(e);
}


