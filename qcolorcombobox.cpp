#include <QColorDialog>
#include "qcolorcombobox.h"

QColor QColorComboBox::color() const
 {
    return qVariantValue<QColor>(itemData(currentIndex(), Qt::DecorationRole));
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
    QPalette newPalette = palette();
    newPalette.setColor(QPalette::Button, color);
    setPalette(newPalette);
}
void QColorToolButton::mouseReleaseEvent ( QMouseEvent * e )
{
    setColor(QColorDialog::getColor(selected_color,
                (QWidget*)parent(),"Select color",QColorDialog::ShowAlphaChannel));
    QToolButton::mouseReleaseEvent(e);
}


