import { createRequire } from "node:module";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [htmlPath, pdfPath] = process.argv.slice(2);
if (!htmlPath || !pdfPath) {
  throw new Error("Usage: render_urdu_pdf.mjs INPUT.html OUTPUT.pdf");
}

const executablePath = process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const browser = await chromium.launch({
  executablePath,
  headless: true,
  args: ["--disable-dev-shm-usage"],
});
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({
    path: pdfPath,
    width: "6in",
    height: "9in",
    printBackground: true,
    preferCSSPageSize: true,
    displayHeaderFooter: true,
    margin: { top: "0.55in", right: "0.52in", bottom: "0.55in", left: "0.52in" },
    headerTemplate: '<div style="width:100%;padding:0 0.52in;font:8px Georgia,serif;color:#54605a;text-align:center">Quran - Evidence-Audited Modern Urdu Translation</div>',
    footerTemplate: '<div style="width:100%;font:8px Georgia,serif;color:#54605a;text-align:center"><span class="pageNumber"></span></div>',
  });
} finally {
  await browser.close();
}
