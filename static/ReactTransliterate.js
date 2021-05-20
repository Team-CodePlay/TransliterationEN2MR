const e = React.createElement;
function _interopDefault(ex) {
  return ex && typeof ex === "object" && "default" in ex ? ex["default"] : ex;
}

/* jshint browser: true */

(function () {
  // We'll copy the properties below into the mirror div.
  // Note that some browsers, such as Firefox, do not concatenate properties
  // into their shorthand (e.g. padding-top, padding-bottom etc. -> padding),
  // so we have to list every single property explicitly.
  var properties = [
    "direction", // RTL support
    "boxSizing",
    "width", // on Chrome and IE, exclude the scrollbar, so the mirror div wraps exactly as the textarea does
    "height",
    "overflowX",
    "overflowY", // copy the scrollbar for IE

    "borderTopWidth",
    "borderRightWidth",
    "borderBottomWidth",
    "borderLeftWidth",
    "borderStyle",

    "paddingTop",
    "paddingRight",
    "paddingBottom",
    "paddingLeft",

    // https://developer.mozilla.org/en-US/docs/Web/CSS/font
    "fontStyle",
    "fontVariant",
    "fontWeight",
    "fontStretch",
    "fontSize",
    "fontSizeAdjust",
    "lineHeight",
    "fontFamily",

    "textAlign",
    "textTransform",
    "textIndent",
    "textDecoration", // might not make a difference, but better be safe

    "letterSpacing",
    "wordSpacing",

    "tabSize",
    "MozTabSize",
  ];

  var isBrowser = typeof window !== "undefined";
  var isFirefox = isBrowser && window.mozInnerScreenX != null;

  function getCaretCoordinates(element, position, options) {
    if (!isBrowser) {
      throw new Error(
        "textarea-caret-position#getCaretCoordinates should only be called in a browser"
      );
    }

    var debug = (options && options.debug) || false;
    if (debug) {
      var el = document.querySelector(
        "#input-textarea-caret-position-mirror-div"
      );
      if (el) el.parentNode.removeChild(el);
    }

    // The mirror div will replicate the textarea's style
    var div = document.createElement("div");
    div.id = "input-textarea-caret-position-mirror-div";
    document.body.appendChild(div);

    var style = div.style;
    var computed = window.getComputedStyle
      ? window.getComputedStyle(element)
      : element.currentStyle; // currentStyle for IE < 9
    var isInput = element.nodeName === "INPUT";

    // Default textarea styles
    style.whiteSpace = "pre-wrap";
    if (!isInput) style.wordWrap = "break-word"; // only for textarea-s

    // Position off-screen
    style.position = "absolute"; // required to return coordinates properly
    if (!debug) style.visibility = "hidden"; // not 'display: none' because we want rendering

    // Transfer the element's properties to the div
    properties.forEach(function (prop) {
      if (isInput && prop === "lineHeight") {
        // Special case for <input>s because text is rendered centered and line height may be != height
        if (computed.boxSizing === "border-box") {
          var height = parseInt(computed.height);
          var outerHeight =
            parseInt(computed.paddingTop) +
            parseInt(computed.paddingBottom) +
            parseInt(computed.borderTopWidth) +
            parseInt(computed.borderBottomWidth);
          var targetHeight = outerHeight + parseInt(computed.lineHeight);
          if (height > targetHeight) {
            style.lineHeight = height - outerHeight + "px";
          } else if (height === targetHeight) {
            style.lineHeight = computed.lineHeight;
          } else {
            style.lineHeight = 0;
          }
        } else {
          style.lineHeight = computed.height;
        }
      } else {
        style[prop] = computed[prop];
      }
    });

    if (isFirefox) {
      // Firefox lies about the overflow property for textareas: https://bugzilla.mozilla.org/show_bug.cgi?id=984275
      if (element.scrollHeight > parseInt(computed.height))
        style.overflowY = "scroll";
    } else {
      style.overflow = "hidden"; // for Chrome to not render a scrollbar; IE keeps overflowY = 'scroll'
    }

    div.textContent = element.value.substring(0, position);
    // The second special handling for input type="text" vs textarea:
    // spaces need to be replaced with non-breaking spaces - http://stackoverflow.com/a/13402035/1269037
    if (isInput) div.textContent = div.textContent.replace(/\s/g, "\u00a0");

    var span = document.createElement("span");
    // Wrapping must be replicated *exactly*, including when a long word gets
    // onto the next line, with whitespace at the end of the line before (#7).
    // The  *only* reliable way to do that is to copy the *entire* rest of the
    // textarea's content into the <span> created at the caret position.
    // For inputs, just '.' would be enough, but no need to bother.
    span.textContent = element.value.substring(position) || "."; // || because a completely empty faux span doesn't render at all
    div.appendChild(span);

    var coordinates = {
      top: span.offsetTop + parseInt(computed["borderTopWidth"]),
      left: span.offsetLeft + parseInt(computed["borderLeftWidth"]),
      height: parseInt(computed["lineHeight"]),
    };

    if (debug) {
      span.style.backgroundColor = "#aaa";
    } else {
      document.body.removeChild(div);
    }

    return coordinates;
  }

  if (typeof module != "undefined" && typeof module.exports != "undefined") {
    module.exports = getCaretCoordinates;
  } else if (isBrowser) {
    window.getCaretCoordinates = getCaretCoordinates;
  }
})();

function _extends() {
  _extends =
    Object.assign ||
    function (target) {
      for (var i = 1; i < arguments.length; i++) {
        var source = arguments[i];

        for (var key in source) {
          if (Object.prototype.hasOwnProperty.call(source, key)) {
            target[key] = source[key];
          }
        }
      }

      return target;
    };

  return _extends.apply(this, arguments);
}

function _objectWithoutPropertiesLoose(source, excluded) {
  if (source == null) return {};
  var target = {};
  var sourceKeys = Object.keys(source);
  var key, i;

  for (i = 0; i < sourceKeys.length; i++) {
    key = sourceKeys[i];
    if (excluded.indexOf(key) >= 0) continue;
    target[key] = source[key];
  }

  return target;
}

// A type of promise-like that resolves synchronously and supports only one observer

const _iteratorSymbol =
  /*#__PURE__*/ typeof Symbol !== "undefined"
    ? Symbol.iterator || (Symbol.iterator = Symbol("Symbol.iterator"))
    : "@@iterator";

const _asyncIteratorSymbol =
  /*#__PURE__*/ typeof Symbol !== "undefined"
    ? Symbol.asyncIterator ||
      (Symbol.asyncIterator = Symbol("Symbol.asyncIterator"))
    : "@@asyncIterator";

// Asynchronously call a function and send errors to recovery continuation
function _catch(body, recover) {
  try {
    var result = body();
  } catch (e) {
    return recover(e);
  }
  if (result && result.then) {
    return result.then(void 0, recover);
  }
  return result;
}

function isTouchEnabled() {
  return (
    "ontouchstart" in window ||
    navigator.maxTouchPoints > 0 ||
    navigator.msMaxTouchPoints > 0
  );
}

function getInputSelection(el) {
  var start = 0;
  var end = 0;

  if (!el) {
    return {
      start: start,
      end: end,
    };
  }

  if (
    typeof el.selectionStart === "number" &&
    typeof el.selectionEnd === "number"
  ) {
    return {
      start: el.selectionStart,
      end: el.selectionEnd,
    };
  }

  return {
    start: start,
    end: end,
  };
}
function setCaretPosition(elem, caretPos) {
  if (elem) {
    if (elem.selectionStart) {
      elem.focus();
      elem.setSelectionRange(caretPos, caretPos);
    } else {
      elem.focus();
    }
  }
}

var classes = { ReactTransliterate: "_1tkHS", Active: "_1KtfG" };

var TriggerKeys = {
  KEY_RETURN: 13,
  KEY_ENTER: 14,
  KEY_TAB: 9,
  KEY_SPACE: 32,
};

var KEY_UP = 38;
var KEY_DOWN = 40;
var KEY_ESCAPE = 27;
var OPTION_LIST_Y_OFFSET = 10;
var OPTION_LIST_MIN_WIDTH = 100;
var ReactTransliterate = function ReactTransliterate(_ref) {
  var _ref$renderComponent = _ref.renderComponent,
    renderComponent =
      _ref$renderComponent === void 0
        ? function (props) {
            return React.createElement("input", Object.assign({}, props));
          }
        : _ref$renderComponent,
    _ref$lang = _ref.lang,
    lang = _ref$lang === void 0 ? "hi" : _ref$lang,
    _ref$offsetX = _ref.offsetX,
    offsetX = _ref$offsetX === void 0 ? 0 : _ref$offsetX,
    _ref$offsetY = _ref.offsetY,
    offsetY = _ref$offsetY === void 0 ? 10 : _ref$offsetY,
    _ref$onChange = _ref.onChange,
    onChange = _ref$onChange === void 0 ? function () {} : _ref$onChange,
    _ref$onChangeText = _ref.onChangeText,
    onChangeText =
      _ref$onChangeText === void 0 ? function () {} : _ref$onChangeText,
    _ref$onBlur = _ref.onBlur,
    onBlur = _ref$onBlur === void 0 ? function () {} : _ref$onBlur,
    value = _ref.value,
    _ref$onKeyDown = _ref.onKeyDown,
    onKeyDown = _ref$onKeyDown === void 0 ? function () {} : _ref$onKeyDown,
    _ref$containerClassNa = _ref.containerClassName,
    containerClassName =
      _ref$containerClassNa === void 0 ? "" : _ref$containerClassNa,
    _ref$containerStyles = _ref.containerStyles,
    containerStyles =
      _ref$containerStyles === void 0 ? {} : _ref$containerStyles,
    _ref$activeItemStyles = _ref.activeItemStyles,
    activeItemStyles =
      _ref$activeItemStyles === void 0 ? {} : _ref$activeItemStyles,
    _ref$maxOptions = _ref.maxOptions,
    maxOptions = _ref$maxOptions === void 0 ? 5 : _ref$maxOptions,
    _ref$hideSuggestionBo = _ref.hideSuggestionBoxOnMobileDevices,
    hideSuggestionBoxOnMobileDevices =
      _ref$hideSuggestionBo === void 0 ? false : _ref$hideSuggestionBo,
    _ref$hideSuggestionBo2 = _ref.hideSuggestionBoxBreakpoint,
    hideSuggestionBoxBreakpoint =
      _ref$hideSuggestionBo2 === void 0 ? 450 : _ref$hideSuggestionBo2,
    _ref$triggerKeys = _ref.triggerKeys,
    triggerKeys =
      _ref$triggerKeys === void 0
        ? [
            TriggerKeys.KEY_SPACE,
            TriggerKeys.KEY_ENTER,
            TriggerKeys.KEY_RETURN,
            TriggerKeys.KEY_TAB,
          ]
        : _ref$triggerKeys,
    _ref$insertCurrentSel = _ref.insertCurrentSelectionOnBlur,
    insertCurrentSelectionOnBlur =
      _ref$insertCurrentSel === void 0 ? true : _ref$insertCurrentSel,
    _ref$showCurrentWordA = _ref.showCurrentWordAsLastSuggestion,
    showCurrentWordAsLastSuggestion =
      _ref$showCurrentWordA === void 0 ? true : _ref$showCurrentWordA,
    rest = _objectWithoutPropertiesLoose(_ref, [
      "renderComponent",
      "lang",
      "offsetX",
      "offsetY",
      "onChange",
      "onChangeText",
      "onBlur",
      "value",
      "onKeyDown",
      "containerClassName",
      "containerStyles",
      "activeItemStyles",
      "maxOptions",
      "hideSuggestionBoxOnMobileDevices",
      "hideSuggestionBoxBreakpoint",
      "triggerKeys",
      "insertCurrentSelectionOnBlur",
      "showCurrentWordAsLastSuggestion",
    ]);

  var _useState = React.useState([]),
    options = _useState[0],
    setOptions = _useState[1];

  var _useState2 = React.useState(0),
    left = _useState2[0],
    setLeft = _useState2[1];

  var _useState3 = React.useState(0),
    top = _useState3[0],
    setTop = _useState3[1];

  var _useState4 = React.useState(0),
    selection = _useState4[0],
    setSelection = _useState4[1];

  var _useState5 = React.useState(-1),
    matchStart = _useState5[0],
    setMatchStart = _useState5[1];

  var _useState6 = React.useState(-1),
    matchEnd = _useState6[0],
    setMatchEnd = _useState6[1];

  var inputRef = React.useRef(null);

  var _useState7 = React.useState({
      width: 0,
      height: 0,
    }),
    windowSize = _useState7[0],
    setWindowSize = _useState7[1];

  var shouldRenderSuggestions = React.useMemo(
    function () {
      return hideSuggestionBoxOnMobileDevices
        ? windowSize.width > hideSuggestionBoxBreakpoint
        : true;
    },
    [windowSize, hideSuggestionBoxBreakpoint, hideSuggestionBoxOnMobileDevices]
  );

  var reset = function reset() {
    setSelection(0);
    setOptions([]);
  };

  var handleSelection = function handleSelection(index) {
    var _inputRef$current;

    var currentString = value;
    if (typeof currentString !== "string") return;
    var newValue =
      currentString.substring(0, matchStart) +
      options[index] +
      " " +
      currentString.substring(matchEnd + 1, currentString.length);
    setTimeout(function () {
      setCaretPosition(
        inputRef.current,
        matchStart + options[index].length + 1
      );
    }, 1);
    var e = {
      target: {
        value: newValue,
      },
    };
    onChangeText(newValue);
    onChange(e);
    reset();
    return (_inputRef$current = inputRef.current) === null ||
      _inputRef$current === void 0
      ? void 0
      : _inputRef$current.focus();
  };

  var getSuggestions = function getSuggestions(lastWord) {
    try {
      if (!shouldRenderSuggestions) {
        return Promise.resolve();
      }

      var numOptions = showCurrentWordAsLastSuggestion
        ? maxOptions - 1
        : maxOptions;
      var url = "http://127.0.0.1:5000/transliterate";
      var payload = { input: lastWord };

      var _temp2 = _catch(
        function () {
          return new Promise(function (resolve, reject) {
            $.ajax({
              type: "POST",
              url: url,
              data: payload,
              success: function (result, status) {
                setOptions(result.transliteration);
                resolve(result);
              },
            });
          });
        },
        function (e) {
          console.error("There was an error with transliteration", e);
        }
      );

      return Promise.resolve(
        _temp2 && _temp2.then ? _temp2.then(function () {}) : void 0
      );
    } catch (e) {
      return Promise.reject(e);
    }
  };

  var handleChange = function handleChange(e) {
    var value = e.currentTarget.value;
    onChange(e);
    onChangeText(value);

    if (!shouldRenderSuggestions) {
      return;
    }

    var caret = getInputSelection(e.target).end;
    var input = inputRef.current;
    if (!input) return;
    var caretPos = getCaretCoordinates(input, caret);
    var indexOfLastSpace =
      value.lastIndexOf(" ", caret - 1) < value.lastIndexOf("\n", caret - 1)
        ? value.lastIndexOf("\n", caret - 1)
        : value.lastIndexOf(" ", caret - 1);
    setMatchStart(indexOfLastSpace + 1);
    setMatchEnd(caret - 1);
    var currentWord = value.slice(indexOfLastSpace + 1, caret);

    if (currentWord) {
      getSuggestions(currentWord);
      var rect = input.getBoundingClientRect();

      var _top =
        caretPos.top < rect.height
          ? caretPos.top + input.offsetTop
          : rect.height -
            ((input.scrollHeight - caretPos.top) % rect.height) +
            input.offsetTop;

      var _left = Math.min(
        caretPos.left + input.offsetLeft - OPTION_LIST_Y_OFFSET,
        input.offsetLeft + rect.width - OPTION_LIST_MIN_WIDTH
      );

      setTop(_top);
      setLeft(_left);
    } else {
      reset();
    }
  };

  var handleKeyDown = function handleKeyDown(event) {
    var helperVisible = options.length > 0;

    if (helperVisible) {
      if (triggerKeys.includes(event.keyCode)) {
        event.preventDefault();
        handleSelection(selection);
      } else {
        switch (event.keyCode) {
          case KEY_ESCAPE:
            event.preventDefault();
            reset();
            break;

          case KEY_UP:
            event.preventDefault();
            setSelection((options.length + selection - 1) % options.length);
            break;

          case KEY_DOWN:
            event.preventDefault();
            setSelection((selection + 1) % options.length);
            break;

          default:
            onKeyDown(event);
            break;
        }
      }
    } else {
      onKeyDown(event);
    }
  };

  var handleBlur = function handleBlur(event) {
    if (!isTouchEnabled()) {
      if (insertCurrentSelectionOnBlur && options[0]) {
        handleSelection(0);
      } else {
        reset();
      }
    }

    onBlur(event);
  };

  var handleResize = function handleResize() {
    var width = window.innerWidth;
    var height = window.innerHeight;
    setWindowSize({
      width: width,
      height: height,
    });
  };

  React.useEffect(function () {
    window.addEventListener("resize", handleResize);
    var width = window.innerWidth;
    var height = window.innerHeight;
    setWindowSize({
      width: width,
      height: height,
    });
    return function () {
      window.removeEventListener("resize", handleResize);
    };
  }, []);
  return React.createElement(
    "div",
    {
      style: _extends({}, containerStyles, {
        position: "relative",
      }),
      className: containerClassName,
    },
    renderComponent(
      _extends(
        {
          onChange: handleChange,
          onKeyDown: handleKeyDown,
          onBlur: handleBlur,
          ref: inputRef,
          value: value,
        },
        rest
      )
    ),
    shouldRenderSuggestions &&
      options.length > 0 &&
      React.createElement(
        "ul",
        {
          style: {
            left: left + offsetX + "px",
            top: top + offsetY + "px",
            position: "absolute",
            width: "auto",
          },
          className: classes.ReactTransliterate,
        },
        options.map(function (item, index) {
          return React.createElement(
            "li",
            {
              className: index === selection ? classes.Active : undefined,
              style: index === selection ? activeItemStyles || {} : {},
              onMouseEnter: function onMouseEnter() {
                setSelection(index);
              },
              onClick: function onClick() {
                return handleSelection(index);
              },
              key: item,
            },
            item
          );
        })
      )
  );
};

//exports.ReactTransliterate = ReactTransliterate;
//exports.TriggerKeys = TriggerKeys;
//# sourceMappingURL=index.js.map
