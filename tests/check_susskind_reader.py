#!/usr/bin/env python3
"""Exercise multilingual controls and real PDF.js rendering on desktop/mobile."""
import argparse
import json
import mimetypes
from pathlib import Path
from urllib.parse import unquote, urlparse

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SUSSKIND = ROOT.parent / 'leonardsusskind'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use published files without local overrides')
    parser.add_argument('--evidence', type=Path, default=SUSSKIND / '.translation-work/evidence/lazylearn')
    args = parser.parse_args()
    args.evidence.mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        try:
            for name, width, height in [('desktop', 1280, 900), ('mobile', 390, 844)]:
                context = browser.new_context(viewport={'width': width, 'height': height}, service_workers='block')
                if not args.live:
                    def route(request):
                        url = urlparse(request.request.url)
                        root = None
                        if url.hostname == 'learn.lazying.art':
                            root, rel = ROOT / 'docs', unquote(url.path).lstrip('/')
                        elif url.hostname == 'raw.githubusercontent.com' and url.path.startswith('/lachlanchen/leonardsusskind/main/'):
                            root, rel = SUSSKIND, unquote(url.path).split('/main/', 1)[1]
                        if root:
                            path = (root / rel).resolve()
                            if path.is_relative_to(root) and path.is_file():
                                request.fulfill(path=str(path), content_type=mimetypes.guess_type(path)[0] or 'application/octet-stream',
                                                headers={'Access-Control-Allow-Origin': '*'})
                                return
                        request.continue_()
                    context.route('**/*', route)
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('console', lambda message: errors.append(message.text) if message.type == 'error' else None)
                page.goto('https://learn.lazying.art/leonardsusskind-reader.html?book=demystifying-the-higgs-boson&lang=ja&utm_source=qa', wait_until='domcontentloaded')
                languages = page.locator('#language-select option').evaluate_all('(items) => items.map(x => x.value)')
                assert languages == ['en', 'ja', 'zh'], languages
                for lang in ['ja', 'zh']:
                    if lang != 'ja':
                        page.select_option('#language-select', lang)
                        page.wait_for_url('**lang=zh**')
                    assert page.locator('#language-select').input_value() == lang
                    assert f'/{lang}/book.pdf' in page.locator('#raw-pdf-link').get_attribute('href')
                    assert 'utm_source=qa' in page.url
                    viewer = page.frame_locator('#pdf-frame')
                    page.locator('#pdf-frame').scroll_into_view_if_needed()
                    try:
                        viewer.locator('.page[data-page-number="1"] canvas').wait_for(timeout=90000)
                    except Exception:
                        page.screenshot(path=str(args.evidence / f'{name}-{lang}-failed.png'))
                        print(json.dumps({'errors': errors, 'frames': [frame.url for frame in page.frames]}), flush=True)
                        raise
                    assert viewer.locator('#outerContainer.sidebarOpen').count() == 0
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth + 2')
                    page.locator('.pdf-reader-toolbar').scroll_into_view_if_needed()
                    page.screenshot(path=str(args.evidence / f'{name}-{lang}.png'))
                    results.append({'viewport': name, 'language': lang, 'pdfCanvasRendered': True, 'pageOverflow': False})
                context.close()
        finally:
            browser.close()
    (args.evidence / 'audit.json').write_text(json.dumps({'live': args.live, 'checks': results}, indent=2) + '\n')
    print(json.dumps(results))


if __name__ == '__main__':
    main()
