######################################################################
# Customized qmake compilation settings
# Last modified: 2017.04.28 - Jake
# Author: Jake Retallick
######################################################################


# CONFIG-=debug		# uncomment to exclude debugging symbols

QT += core gui widgets

greaterThan(QT_MAJOR_VERSION, 5): QT += widgets

TEMPLATE = app
TARGET = db-sim
INCLUDEPATH += .

RESOURCES = resources/application.qrc

# Input GUI Headers and SOURCE
HEADERS += \
	src/gui/application.h \
	src/settings/settings.h \
	src/gui/widgets/design_panel.h \
	src/gui/widgets/dialog_panel.h \
	src/gui/widgets/input_field.h \
	src/gui/widgets/info_panel.h \
	src/gui/widgets/lattice.h \
	src/gui/widgets/primitives/emitter.h \
	src/gui/widgets/primitives/items.h \
	src/gui/widgets/primitives/layer.h \
	src/gui/widgets/primitives/dbdot.h \
	src/gui/widgets/primitives/ghost.h

SOURCES += \
	src/main.cpp \
	src/gui/application.cpp \
	src/settings/settings.cpp \
	src/gui/widgets/design_panel.cpp \
	src/gui/widgets/dialog_panel.cpp \
	src/gui/widgets/input_field.cpp \
	src/gui/widgets/info_panel.cpp \
	src/gui/widgets/lattice.cpp \
	src/gui/widgets/primitives/emitter.cpp \
	src/gui/widgets/primitives/items.cpp \
	src/gui/widgets/primitives/layer.cpp \
	src/gui/widgets/primitives/dbdot.cpp \
	src/gui/widgets/primitives/ghost.cpp


#HEADERS += \
#	src/engines/core/constants.h \
#	src/engines/core/common.h \
#	src/engines/core/classes.h \
#	src/engines/core/problem.h \
#	src/engines/core/base_engine.h

#SOURCES += \
#	src/engines/core/common.cpp \
#	src/engines/core/problem.cpp \
#	src/engines/core/base_engine.cpp


#####################
# BUILD DIRECTORIES #
#####################

release:	DESTDIR = build/release
debug:		DESTDIR = build/debug

OBJECTS_DIR	= $$DESTDIR/.obj
MOC_DIR		= $$DESTDIR/.moc
RCC_DIR		= $$DESTDIR/.qrc
UI_DIR		= $$DESTDIR/.ui
