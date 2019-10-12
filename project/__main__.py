from .app import init_logging, init_controller, init_ocr, init_ui, init_model, init_cleanup, wait_for_init, mainloop

def main():
    init_logging()
    nn         = init_model()
    ocr        = init_ocr()
    qwop, rect = init_controller()
    ui         = init_ui(rect)
    init_cleanup(qwop, ocr, ui, nn)
    wait_for_init()
    mainloop(qwop, ocr, ui, nn)

if __name__ == '__main__':
    main()
