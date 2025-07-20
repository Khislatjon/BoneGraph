//
//  ArxivXMLParser.swift
//  PaperScraper
//
//  Created by Khislatjon Valijonov on 20/07/2025.
//

import Foundation

class ArxivXMLParser: NSObject, XMLParserDelegate {
    private var papers: [ArxivPaper] = []
    private var currentElement = ""
    private var currentTitle = ""
    private var currentSummary = ""
    private var currentLink = ""
    private var currentAuthors: [String] = []
    private var currentAuthorName = ""
    private var insideEntry = false
    private var currentPublishedDate = ""

    func parse(data: Data) -> [ArxivPaper] {
        let parser = XMLParser(data: data)
        parser.delegate = self
        parser.parse()
        return papers
    }

    func parser(_ parser: XMLParser, didStartElement elementName: String, namespaceURI: String?, qualifiedName qName: String?, attributes attributeDict: [String : String] = [:]) {
        currentElement = elementName
        if elementName == "entry" {
            insideEntry = true
            currentTitle = ""
            currentSummary = ""
            currentLink = ""
            currentAuthors = []
            currentPublishedDate = ""
        } else if elementName == "link", let href = attributeDict["href"], attributeDict["rel"] == "alternate" {
            currentLink = href
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard insideEntry else { return }
        switch currentElement {
        case "title":
            currentTitle += string
        case "summary":
            currentSummary += string
        case "name":
            currentAuthorName += string
        case "published":
            currentPublishedDate += string
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, didEndElement elementName: String, namespaceURI: String?, qualifiedName qName: String?) {
        if elementName == "entry" {
            let year = parseYear(from: currentPublishedDate)
            let paper = ArxivPaper(
                title: currentTitle.trimmingCharacters(in: .whitespacesAndNewlines),
                summary: currentSummary.trimmingCharacters(in: .whitespacesAndNewlines),
                authors: currentAuthors,
                link: currentLink,
                year: year
            )
            papers.append(paper)
            insideEntry = false
        } else if elementName == "name" {
            currentAuthors.append(currentAuthorName.trimmingCharacters(in: .whitespacesAndNewlines))
            currentAuthorName = ""
        }
    }
    
    private func parseYear(from publishedDate: String) -> Int {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: publishedDate) {
            let calendar = Calendar(identifier: .gregorian)
            return calendar.component(.year, from: date)
        }
        return -1 // fallback if parsing fails
    }
}
